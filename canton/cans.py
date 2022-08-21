# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 
import numpy as np
import time

from .misc import *

# this library is Python 3 only.

# this library aims to wrap up the ugliness of tensorflow,
# at the same time provide a better interface for NON-STANDARD
# learning experiments(such as GANs, etc.) than Keras.

# a Can is a container. it can contain other Cans.

class Can:
    def __init__(self):
        self.subcans = [] # other cans contained
        self.weights = [] # trainable
        self.biases = []
        self.only_weights = []
        self.variables = [] # should save with the weights, but not trainable
        self.updates = [] # update ops, mainly useful for batch norm
        # well, you decide which one to put into

        self.inference = None

    # by making weight, you create trainable variables
    def make_weight(self,shape,name='W', mean=0., stddev=1e-2, initializer=None):
        mean,stddev = [float(k) for k in [mean,stddev]]
        if initializer is None:
            initial = tf.random.truncated_normal(shape, mean=mean, stddev=stddev)
        else:
            initial = initializer
        w = tf.Variable(initial,name=name)
        self.weights.append(w)
        self.only_weights.append(w)
        return w

    def make_bias(self,shape,name='b', mean=0.):
        mean = float(mean)
        initial = tf.constant(mean, shape=shape)
        b = tf.Variable(initial,name=name)
        self.weights.append(b)
        self.biases.append(b)
        return b

    # make a variable that is not trainable, by passing in a numpy array
    def make_variable(self,nparray,name='v'):
        v = tf.Variable(nparray,name=name)
        self.variables.append(v)
        return v

    # add an op as update op of this can
    def make_update(self,op):
        self.updates.append(op)
        return op

    # put other cans inside this can, as subcans
    def incan(self,c):
        if hasattr(c,'__iter__'): # if iterable
            self.subcans += list(c)
        else:
            self.subcans += [c]
        # return self

    # another name for incan
    def add(self,c):
        self.incan(c)
        return c

    # if you don't wanna specify the __call__ function manually,
    # you may chain up all the subcans to make one:
    def chain(self):
        def call(i):
            for c in self.subcans:
                i = c(i)
            return i
        self.set_function(call)

    # traverse the tree of all subcans,
    # and extract a flattened list of certain attributes.
    # the attribute itself should be a list, such as 'weights'.
    # f is the transformer function, applied to every entry
    def traverse(self,target='weights',f=lambda x:x):
        l = [f(a) for a in getattr(self,target)] + [c.traverse(target,f) for c in self.subcans]
        # the flatten logic is a little bit dirty
        return list(flatten(l, lambda x:isinstance(x,list)))

    # return weight tensors of current can and it's subcans
    def get_weights(self):
        return self.traverse('weights')

    def get_biases(self):
        return self.traverse('biases')

    def get_only_weights(self): # dont get biases
        return self.traverse('only_weights')

    # return update operations of current can and it's subcans
    def get_updates(self):
        return self.traverse('updates')

    # set __call__ function
    def set_function(self,func):
        self.func = func

    # default __call__
    def __call__(self,i,*args,**kwargs):
        if hasattr(self,'func'):
            return self.func(i,*args,**kwargs)
        else:
            raise NameError('You didnt override __call__(), nor called set_function()/chain()')

    def get_value_of(self,tensors):
        sess = get_session()
        values = sess.run(tensors)
        return values

    def save_weights(self,filename): # save both weights and variables
        with open(filename,'wb') as f:
            # extract all weights in one go:
            w = self.get_value_of(self.get_weights()+self.traverse('variables'))
            print(len(w),'weights (and variables) obtained.')

            # create an array object and put all the arrays into it.
            # otherwise np.asanyarray() within np.savez_compressed()
            # might make stupid mistakes
            arrobj = np.empty([len(w)],dtype='object') # array object
            for i in range(len(w)):
                arrobj[i] = w[i]

            np.savez_compressed(f,w=arrobj)
            print('successfully saved to',filename)
            return True

    def load_weights(self,filename):
        with open(filename,'rb') as f:
            loaded_w = np.load(f, allow_pickle=True)
            print('successfully loaded from',filename)
            if hasattr(loaded_w,'items'):
                # compressed npz (newer)
                loaded_w = loaded_w['w']
            else:
                # npy (older)
                pass
            # but we cannot assign all those weights in one go...
            model_w = self.get_weights()+self.traverse('variables')
            if len(loaded_w)!=len(model_w):
                raise NameError('number of weights (variables) from the file({}) differ from the model({}).'.format(len(loaded_w),len(model_w)))
            else:
                assign_ops = [tf.compat.v1.assign(model_w[i],loaded_w[i])
                    for i,_ in enumerate(model_w)]

            sess = get_session()
            sess.run(assign_ops)
            print(len(loaded_w),'weights assigned.')
            return True

    def infer(self,i):
        # run function, return value
        if self.inference is None:
            # the inference graph will be created when you infer for the first time
            # 1. create placeholders with same dimensions as the input
            if isinstance(i,list): # if Can accept more than one input
                x = [tf.placeholder(tf.float32,shape=[None for _ in range(len(j.shape))])
                    for j in i]
                print('(infer) input is list.')
            else:
                x = tf.placeholder(tf.float32, shape=[None for _ in range(len(i.shape))])

            # 2. set training state to false, construct the graph
            set_training_state(False)
            y = self.__call__(x)
            set_training_state(True)

            # 3. create the inference function
            def inference(k):
                sess = get_session()
                if isinstance(i,list):
                    res = sess.run([y],feed_dict={x[j]:k[j]
                        for j,_ in enumerate(x)})[0]
                else:
                    res = sess.run([y],feed_dict={x:k})[0]
                return res
            self.inference = inference

        return self.inference(i)

    def summary(self):
        print('-------------------')
        print('Directly Trainable:')
        variables_summary(self.get_weights())
        print('-------------------')
        print('Not Directly Trainable:')
        variables_summary(self.traverse('variables'))
        print('-------------------')

def variables_summary(var_list):
    shapes = [v.get_shape() for v in var_list]
    shape_lists = [s.as_list() for s in shapes]
    shape_lists = list(map(lambda x:''.join(map(lambda x:'{:>5}'.format(x),x)),shape_lists))

    num_elements = [s.num_elements() for s in shapes]
    total_num_of_variables = sum(num_elements)
    names = [v.name for v in var_list]

    print('counting variables...')
    for i in range(len(shapes)):
        print('{:>25}  ->  {:<6} {}'.format(
        shape_lists[i],num_elements[i],names[i]))

    print('{:>25}  ->  {:<6} {}'.format(
    'tensors: '+str(len(shapes)),
    str(total_num_of_variables),
    'variables'))

# you know, MLP
class Dense(Can):
    def __init__(self,num_inputs,num_outputs,bias=True,mean=None, stddev=None, initializer=None):
        super().__init__()
        # for different output unit type, use different noise scales
        if stddev is None:
            stddev = 2. # 2 for ReLU, 1 for linear/tanh
        stddev = np.sqrt(stddev/num_inputs)
        if mean is None: # mean for bias layer
            mean = 0.

        self.W = self.make_weight([num_inputs,num_outputs],stddev=stddev,initializer=initializer)
        self.use_bias = bias
        if bias:
            self.b = self.make_bias([num_outputs],mean=mean)
    def __call__(self,i):
        d = tf.matmul(i,self.W)
        if self.use_bias:
            return d + self.b
        else:
            return d

class LayerNormDense(Dense):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        nop = args[1]
        self.layernorm = self.add(LayerNorm(nop))

    def __call__(self,i):
        d = tf.matmul(i,self.W)
        d = self.layernorm(d)
        if self.use_bias:
            return d + self.b
        else:
            return d

# apply Dense on last dimension of 1d input. equivalent to 1x1 conv1d.
class TimeDistributedDense(Dense):
    def __call__(self,i):
        s = tf.shape(i)
        b,t,d = s[0],s[1],s[2]
        i = tf.reshape(i,[b*t,d])
        i = super().__call__(i)
        d = tf.shape(i)[1]
        i = tf.reshape(i,[b,t,d])
        return i

# apply Dense on last dimension of Nd input.
class LastDimDense(Dense):
    def __call__(self,i):
        s = tf.shape(i)
        rank = tf.rank(i)
        fore = s[0:rank-1] # foremost dimensions
        last = s[rank-1] # last dimension
        prod = tf.reduce_prod(fore)
        i = tf.reshape(i,[prod,last]) # shape into
        i = super().__call__(i) # call Dense layer
        d = tf.shape(i)[1]
        i = tf.reshape(i,tf.concat([fore,[d]],axis=0)) # shape back
        return i

# expand last dimension by a branching factor. Expect input of shape [Batch Dims]
class Expansion(Can):
    def __init__(self,nip,factor,stddev=1):
        super().__init__()
        self.nip = nip; self.factor = factor; self.nop = nip*factor
        self.W = self.make_weight([nip,factor],stddev=stddev)
        self.b = self.make_bias([nip*factor])

    def __call__(self,i):
        # input: [Batch Dimin] weight: [Dimin Factor] output: [Batch Dimin Factor]
        result = tf.einsum('bi,if->bif', i, self.W)
        result = tf.reshape(result,[-1,self.nop])
        return result + self.b

# you know, shorthand
class Lambda(Can):
    def __init__(self,f):
        super().__init__()
        self.set_function(f)

# you know, to fit
class Reshape(Can):
    def __init__(self,shape):
        super().__init__()
        self.shape = shape
    def __call__(self,i):
        bs = tf.shape(i)[0] # assume [batch, dims...]
        return tf.reshape(i,[bs]+self.shape)

# you know, nonlinearities
class Act(Can):
    def __init__(self,name,alpha=0.2):
        super().__init__()
        def lrelu(i): # fast impl. with only 1 relu
            negative = tf.nn.relu(-i)
            res = i + negative * (1.0-alpha)
            return res
        def lrelu(i):
            return tf.nn.leaky_relu(i, alpha)

        def selu(x):
            # https://arxiv.org/pdf/1706.02515.pdf
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

        def swish(x):
            return x*tf.sigmoid(x) # beta=1 case

        activations = {
            'relu':tf.nn.relu,
            'tanh':tf.tanh,
            'sigmoid':tf.sigmoid,
            'softmax':tf.nn.softmax,
            'elu':tf.nn.elu,
            'lrelu':lrelu,
            'softplus':tf.nn.softplus,
            'selu':selu,
            'swish':swish,
            'relu6':tf.nn.relu6,
        }
        self.set_function(activations[name])

# you know, brain damage
class Drop(Can):
    def __init__(self,prob,switch=None):
        super().__init__()
        self.prob = prob
        self.switch = switch
    def __call__(self,i):
        if self.switch is None: # not using a switch variable (recommended)
            if get_training_state():
                return tf.nn.dropout(i, keep_prob=self.prob)
            else:
                return i
        else:
            # use a switch variable
            # (if the memory is so limited that a separate flow not possible)
            return tf.cond(self.switch,
                lambda:tf.nn.dropout(i,keep_prob=self.prob),
                lambda:i)

# you know, Yann LeCun
class Conv2D(Can):
    # nip and nop: input and output planes
    # k: dimension of kernel, 3 for 3x3, 5 for 5x5
    # rate: atrous conv rate, 1 = not, 2 = skip one
    def __init__(self,nip,nop,k,std=1,usebias=True,rate=1,padding='SAME',stddev=None):
        super().__init__()
        if stddev is None:
            stddev = 2. # 2 for ReLU, 1 for linear/tanh
        if rate>1 and std>1:
            raise('atrous rate can\'t also be greater \
                than one when stride is already greater than one.')

        self.nip,self.nop,self.k,self.std,self.usebias,self.padding,self.rate\
        = nip,nop,k,std,usebias,padding,rate

        self.W = self.make_weight([k,k,nip,nop],stddev=np.sqrt(stddev/(nip*k*k)))
        # self.W = self.make_weight([k,k,nip,nop],stddev=np.sqrt(stddev/(nip*k*k)),
        #     initializer = tf.contrib.framework.convolutional_delta_orthogonal(
        #         gain = stddev, dtype=tf.float32,
        #     )(shape=[k,k,nip,nop])
        # )
        # assume square window
        if usebias==True:
            self.b =self.make_bias([nop])

    def __call__(self,i):
        if self.rate == 1:
            c = tf.nn.conv2d(i,self.W,
                strides=[1, self.std, self.std, 1],
                padding=self.padding)
        else: #dilated conv
            c = tf.nn.atrous_conv2d(i,self.W,
                rate=self.rate,
                padding=self.padding)

        if self.usebias==True:
            return c + self.b
        else:
            return c

class DepthwiseConv2D(Conv2D):
    def __init__(self, nip, nop, stddev=None,*a,**k):
        if stddev is None:
            stddev = 2. # 2 for ReLU, 1 for linear/tanh
        stddev *= nip # scale for depthwise convolution
        super().__init__(nip=nip, nop=nop, stddev=stddev,*a,**k)
    def __call__(self,i):
        c = tf.nn.depthwise_conv2d(i,self.W,
                strides=[1, self.std, self.std, 1],
                padding=self.padding,
                rate=[self.rate, self.rate],
                )

        if self.usebias==True:
            return c + self.b
        else:
            return c

class GroupConv2D(Can):
    def __init__(self,nip,nop,k,num_groups,*a,**kw):
        super().__init__()
        assert nip % num_groups == 0
        assert nop % num_groups == 0
        self.num_groups = num_groups
        self.nipg = nip//num_groups
        self.nopg = nop//num_groups

        self.groups = [Conv2D(self.nipg, self.nopg, k, *a, **kw) for i in range(num_groups)]
        self.incan(self.groups)

    def __call__(self,i):
        out = []
        for idx, conv in enumerate(self.groups):
            inp = i[:, :, :, idx * self.nipg:(idx+1)*self.nipg]
            out.append(conv(inp))
        return tf.concat(out, axis=-1)

class ChannelShuffle(Can): # shuffle the last dimension
    def __init__(self, nip, num_groups):
        super().__init__()
        assert nip % num_groups == 0
        self.nip = nip
        self.num_groups = num_groups
        self.nipg = nip//num_groups

    def __call__(self, i):
        orig_shape = tf.shape(i)
        reshaped = tf.reshape(i, [-1, self.num_groups, self.nipg])
        transposed = tf.transpose(reshaped, perm=[0,2,1])
        output = tf.reshape(reshaped, orig_shape)
        return output

class ShuffleNet(Can):
    def __init__(self, nip, nop, num_groups):
        super().__init__()
        if nip==nop:
            self.std = 1
        elif nip*2 == nop:
            self.std = 2
        else:
            raise Exception('shufflenet unit accept only nip==nop or nip*2==nop.')

        assert nip % num_groups == 0
        self.num_groups = num_groups
        self.nipg = nip//num_groups

        bottleneck_width = nip//4

        self.gc1 = GroupConv2D(nip, bottleneck_width, k=3, num_groups=num_groups, stddev=2)

        self.cs = ChannelShuffle(bottleneck_width, num_groups=num_groups)
        self.dc = DepthwiseConv2D(bottleneck_width,1, k=3, std=self.std)
        self.gc2 = GroupConv2D(bottleneck_width, nip, k=3, num_groups=num_groups, stddev=1)

        self.incan([self.gc1, self.cs, self.dc, self.gc2])

    def __call__(self, i):
        residual = i

        i = self.gc1(i)
        i = Act('relu')(i)
        i = self.cs(i)
        i = self.dc(i)
        i = self.gc2(i)

        if self.std == 1: # dont grow feature map
            out = residual + i
        else: # grow 2x by concatenation
            residual = AvgPool2D(k=3, std=2)(residual)
            out = tf.concat([residual, i], axis=-1)
        return Act('relu')(out)

# upsampling 2d
class Up2D(Can):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
    def __call__(self,i):
        scale = self.scale
        s = tf.shape(i) # assume NHWC
        newsize = [s[1]*scale,s[2]*scale]
        return tf.image.resize_nearest_neighbor(i, size=newsize, align_corners=None, name=None)

# assume padding == 'SAME'.
class Deconv2D(Conv2D):
    def __call__(self,i):
        if self.usebias == True: i -= self.b

        s = tf.shape(i)
        return tf.nn.conv2d_transpose(
            i,
            self.W,
            output_shape=[s[0], s[1]*self.std, s[2]*self.std, self.nip],
            strides=[1, self.std, self.std, 1],
        )

# you know, recurrency
class Scanner(Can):
    def __init__(self,f):
        super().__init__()
        self.f = f
    def __call__(self,i,starting_state=None, inferred_state_shape=None):
        # previous state is useful when running online.
        if starting_state is None:
            if inferred_state_shape is None:
                print('(Scanner) cannot/didnot infer state_shape. use shape of input[0] instead. please make sure the input to the Scanner has the same last dimension as the function being scanned.')
                initializer = tf.zeros_like(i[0])
            else:
                print('(Scanner) using inferred_state_shape')
                initializer = tf.zeros(inferred_state_shape, tf.float32)
        else:
            initializer = starting_state
        scanned = tf.scan(self.f,i,initializer=initializer)
        return scanned

# deal with batch input.
class BatchScanner(Scanner):
    def __call__(self, i, **kwargs):
        rank = tf.rank(i)
        perm = tf.concat([[1,0],tf.range(2,rank)],axis=0)
        it = tf.transpose(i, perm=perm)
        #[Batch, Seq, Blah, Dim] -> [Seq, Batch, Blah, Dim]

        scanned = super().__call__(it, **kwargs)

        rank = tf.rank(scanned)
        perm = tf.concat([[1,0],tf.range(2,rank)],axis=0)
        scanned = tf.transpose(scanned, perm=perm)
        #[Batch, Seq, Blah, Dim] <- [Seq, Batch, Blah, Dim]
        return scanned

# single forward pass version of GRU. Normally we don't use this directly
class GRU_onepass(Can):
    def __init__(self,num_in,num_h):
        super().__init__()
        # assume input has dimension num_in.
        self.num_in,self.num_h = num_in, num_h
        self.wz = Dense(num_in+num_h,num_h,stddev=1,mean=-1) # forget less
        self.wr = Dense(num_in+num_h,num_h,stddev=1)
        self.w = Dense(num_in+num_h,num_h,stddev=1)
        self.incan([self.wz,self.wr,self.w])
        # http://colah.github.io/posts/2015-08-Understanding-LSTMs/

    def __call__(self,i):
        # assume hidden, input is of shape [batch,num_h] and [batch,num_in]
        hidden = i[0]
        inp = i[1]
        wz,wr,w = self.wz,self.wr,self.w
        dims = tf.rank(inp)
        c = tf.concat([hidden,inp],axis=dims-1)
        z = tf.sigmoid(wz(c))
        r = tf.sigmoid(wr(c))
        h_c = tf.tanh(w(tf.concat([hidden*r,inp],axis=dims-1)))
        h_new = (1-z) * hidden + z * h_c
        return h_new

# GRU2 as reported in *Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks*
# the gates are now only driven by hidden state.
# mod: removed reset gate.
# conclusion 20171220: GRU2(without reset gate) is almost as good as GRU.
class GRU2_onepass(Can):
    def __init__(self,num_in,num_h,double=False):
        super().__init__()
        # assume input has dimension num_in.
        self.num_in,self.num_h,self.rect = num_in, num_h, Act('tanh')
        if double==False:
            self.w = Dense(num_in+num_h,num_h,stddev=1.5)
            self.wz = Dense(num_h,num_h,stddev=1)
        else:
            c = Can()
            c.add(Dense(num_in+num_h,int(num_h/2),stddev=1.5))
            c.add(Act('lrelu'))
            c.add(Dense(int(num_h/2),num_h,stddev=1.5))
            c.chain()
            self.w = c
            c = Can()
            c.add(Dense(num_h,int(num_h/2),stddev=1.5))
            c.add(Act('lrelu'))
            c.add(Dense(int(num_h/2),num_h,stddev=1.5))
            c.chain()
            self.wz = c
        self.incan([self.wz,self.w])

    def __call__(self,i):
        # assume hidden, input is of shape [batch,num_h] and [batch,num_in]
        hidden = i[0]
        inp = i[1]
        wz,w = self.wz,self.w
        # dims = tf.rank(inp)
        z = tf.sigmoid(wz(hidden))
        h_c = self.rect(w(tf.concat([hidden,inp],axis=1)))
        h_new = (1-z) * hidden + z * h_c
        return h_new

# vanilla RNN
class RNN_onepass(Can):
    def __init__(self,num_in,num_h,nonlinearity=Act('tanh'),stddev=1):
        super().__init__()
        # assume input has dimension num_in.
        self.num_in,self.num_h,self.rect = num_in, num_h, nonlinearity
        self.w = Dense(num_in+num_h,num_h,stddev=stddev)
        self.incan([self.w,self.rect])

    def __call__(self,i):
        # assume hidden, input is of shape [batch,num_h] and [batch,num_in]
        hidden = i[0]
        inp = i[1]
        c = tf.concat([hidden,inp],axis=1)
        w = self.w
        h_new = self.rect(w(c))
        return h_new

# same but with LayerNorm-ed Dense layers
class GRU_LN_onepass(GRU_onepass):
    def __init__(self,num_in,num_h):
        Can.__init__(self)
        # assume input has dimension num_in.
        self.num_in,self.num_h = num_in, num_h
        self.wz = LayerNormDense(num_in+num_h,num_h,bias=True)
        self.wr = LayerNormDense(num_in+num_h,num_h,bias=True)
        self.w = LayerNormDense(num_in+num_h,num_h,bias=True)
        self.incan([self.wz,self.wr,self.w])

# single forward pass version of GRUConv2D.
class GRUConv2D_onepass(GRU_onepass): # inherit the __call__ method
    def __init__(self,num_in,num_h,*args,**kwargs):
        Can.__init__(self)
        # assume input has dimension num_in.
        self.num_in,self.num_h = num_in, num_h
        self.wz = Conv2D(num_in+num_h,num_h,usebias=False,*args,**kwargs)
        self.wr = Conv2D(num_in+num_h,num_h,usebias=False,*args,**kwargs)
        self.w = Conv2D(num_in+num_h,num_h,usebias=False,*args,**kwargs)
        self.incan([self.wz,self.wr,self.w])

# RNN Can generator from cells, similar to tf.nn.dynamic_rnn
def rnn_gen(name, one_pass_class):
    class RNN(Can):
        def __init__(self,*args,**kwargs):
            super().__init__()
            self.unit = one_pass_class(*args,**kwargs)
            def f(last_state, new_input):
                return self.unit([last_state, new_input])
            self.bscan = BatchScanner(f)
            self.incan([self.unit,self.bscan])
        def __call__(self,i,state_shaper=None,**kwargs):
            # given input, what should be the shape of the state?
            s = tf.shape(i)
            r = tf.rank(i)
            if state_shaper is not None:
                print('(RNN) inferring state_shape using state_shaper().')
                state_shape = state_shaper(i,self.unit.num_h)
            else:
                print('(RNN) inferring state_shape from input.')
                state_shape = tf.concat([[s[0]],s[2:r-1],[self.unit.num_h]],axis=0)
                # [batch, timesteps, blah, dim]->[batch, blah, hidden_dim]

            return self.bscan(i,inferred_state_shape=state_shape, **kwargs)
    RNN.__name__ = name
    return RNN

# you know, Despicable Me
RNN = rnn_gen('RNN', RNN_onepass)
GRU = rnn_gen('GRU', GRU_onepass)
GRU2 = rnn_gen('GRU2', GRU2_onepass)
GRULN = rnn_gen('GRULN', GRU_LN_onepass)
GRUConv2D = rnn_gen('GRUConv2D', GRUConv2D_onepass)

# you know, LeNet
class AvgPool2D(Can):
    def __init__(self,k,std,padding='SAME'):
        super().__init__()
        self.k,self.std,self.padding = k,std,padding

    def __call__(self,i):
        k,std,padding = self.k,self.std,self.padding
        return tf.nn.avg_pool(i, ksize=[1, k, k, 1],
            strides=[1, std, std, 1], padding=padding)

class MaxPool2D(AvgPool2D):
    def __call__(self,i):
        k,std,padding = self.k,self.std,self.padding
        return tf.nn.max_pool(i, ksize=[1, k, k, 1],
            strides=[1, std, std, 1], padding=padding)

# you know, He Kaiming
class ResConv(Can): # v2
    def __init__(self,nip,nop,std=1,bn=True):
        super().__init__()
        # create the necessary cans:
        nbp = int(max(nip,nop)/4) # bottleneck
        self.direct_sum = (nip==nop and std==1)
        # if no downsampling and feature shrinking

        if self.direct_sum:
            self.convs = \
            [Conv2D(nip,nbp,1,usebias=False),
            Conv2D(nbp,nbp,3,usebias=False),
            Conv2D(nbp,nop,1,usebias=False)]
            self.bns = [BatchNorm(nip),BatchNorm(nbp),BatchNorm(nbp)]
        else:
            self.convs = \
            [Conv2D(nip,nbp,1,std=std,usebias=False),
            Conv2D(nbp,nbp,3,usebias=False),
            Conv2D(nbp,nop,1,usebias=False),
            Conv2D(nip,nop,1,std=std,usebias=False)]
            self.bns = [BatchNorm(nip),BatchNorm(nbp),BatchNorm(nbp)]

        self.incan(self.convs+self.bns) # add those cans into collection

    def __call__(self,i):
        def relu(i):
            return tf.nn.relu(i)

        if self.direct_sum:
            ident = i
            i = relu(self.bns[0](i))
            i = self.convs[0](i)
            i = relu(self.bns[1](i))
            i = self.convs[1](i)
            i = relu(self.bns[2](i))
            i = self.convs[2](i)
            out = ident+i
        else:
            i = relu(self.bns[0](i))
            ident = i
            i = self.convs[0](i)
            i = relu(self.bns[1](i))
            i = self.convs[1](i)
            i = relu(self.bns[2](i))
            i = self.convs[2](i)
            ident = self.convs[3](ident)
            out = ident+i
        return out

class BatchNorm(Can):
    def __init__(self,nip,epsilon=1e-3): # number of input planes/features/channels
        super().__init__()
        params_shape = [nip]
        self.beta = self.make_bias(params_shape,name='beta',mean=0.)
        self.gamma = self.make_bias(params_shape,name='gamma',mean=1.)
        self.moving_mean = self.make_variable(
            tf.constant(0.,shape=params_shape),name='moving_mean')
        self.moving_variance = self.make_variable(
            tf.constant(1.,shape=params_shape),name='moving_variance')

        self.epsilon = epsilon

    def __call__(self,x):
        BN_DECAY = 0.99 # moving average constant
        BN_EPSILON = self.epsilon

        # actual mean and var used:
        if get_training_state()==True:

            x_shape = x.get_shape() # [N,H,W,C]
            #params_shape = x_shape[-1:] # [C]

            axes = list(range(len(x_shape) - 1)) # = range(3) = [0,1,2]
            # axes to reduce mean and variance.
            # here mean and variance is estimated per channel(feature map).

            # reduced mean and var(of activations) of each channel.
            mean, variance = tf.nn.moments(x, axes)

            # use immediate when training(speedup convergence), perform update
            moving_mean = tf.compat.v1.assign(self.moving_mean,
                self.moving_mean*(1.-BN_DECAY) + mean*BN_DECAY)
            moving_variance = tf.compat.v1.assign(self.moving_variance,
                self.moving_variance*(1.-BN_DECAY) + variance*BN_DECAY)

            mean, variance = mean + moving_mean * 1e-10, variance + moving_variance * 1e-10
        else:
            # use average when testing(stabilize), don't perform update
            mean, variance = self.moving_mean, self.moving_variance

        x = tf.nn.batch_normalization(x, mean, variance, self.beta, self.gamma, BN_EPSILON)
        return x

# layer normalization on last axis of input.
class LayerNorm(Can):
    def __init__(self,nop=None):
        super().__init__()
        # learnable
        self.alpha = self.make_bias([],mean=1)
        self.beta = self.make_bias([],mean=0)

    def __call__(self,x): # x -> [N, C]
        # axis = len(x.get_shape())-1
        axis = tf.rank(x)-1
        # reduced mean and var(of activations) of each channel.
        mean, var = tf.nn.moments(x, [axis], keep_dims=True) # of shape [N,1] and [N,1]
        # mean, var = [tf.expand_dims(k, -1) for k in [mean,var]]
        var = tf.maximum(var,1e-7)
        stddev = tf.sqrt(var)
        # apply
        normalized = self.alpha * (x-mean) / stddev + self.beta
        # normalized = (x-mean)/stddev
        return normalized

class InstanceNorm(LayerNorm): # for images
    def __call__(self, x):
        mean, var = tf.nn.moments(x, [1,2], keep_dims=True)
        var = var + 1e-8
        stddev = tf.sqrt(var)
        normalized = self.alpha * (x-mean) / stddev + self.beta
        return normalized
