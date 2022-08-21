import tensorflow as tf
import numpy as np
from .cans import *

# additional cans

def castf32(i):
    return tf.cast(i,tf.float32)

# RBF glimpse
# evaluate RBF functions representing foveal attention mechanism over the input image, given offset.
class Glimpse2D(Can):
    def __init__(self, num_receptors, pixel_span=20):
        super().__init__()
        if num_receptors<1:
            raise NameError('num_receptors should be greater than 0')
        self.num_receptors = nr = num_receptors
        self.pixel_span = ps = pixel_span

        # generate initial positions for receptive fields
        positions = np.zeros((nr,2),dtype='float32')
        w = int(np.ceil(np.sqrt(nr)))
        index = 0
        for row in range(w):
            for col in range(w):
                if index<nr:
                    positions[index,0] = row/(w-1)
                    positions[index,1] = col/(w-1)
                    index+=1
                else:
                    break

        # positions = np.random.uniform(low=-ps/2,high=ps/2,size=(nr,2)).astype('float32')
        positions = (positions - 0.5) * ps * 0.8
        m = tf.Variable(positions,name='means')
        self.weights.append(m)
        self.means = m

        # stddev of receptive fields
        stddevs = (np.ones((nr,1))*ps*0.5*(1/(w-1))).astype('float32')
        s = tf.Variable(stddevs,name='stddevs')
        self.weights.append(s)
        self.stddevs = s

    def shifted_means_given_offsets(self,offsets):
        means = self.means # [num_of_receptor, 2]

        means = tf.expand_dims(means,axis=0) # [1, num_of_receptor, 2]
        offsets = tf.expand_dims(offsets,axis=1) # [batch, 1, 2]

        shifted_means = means + offsets # [batch, num_of_receptor, 2]

        return shifted_means

    def variances(self):
        variances = tf.nn.softplus(self.stddevs)**2 # [num_of_receptor, 1]
        return variances

    def __call__(self,i): # input: [image, offsets]
        offsets = i[1] # offsets [batch, 2]
        images = i[0] # [batch, h, w, c]

        shifted_means =\
            self.shifted_means_given_offsets(offsets)

        variances = self.variances() # [num_of_receptor, 1]
        variances = tf.expand_dims(variances, axis=0)
        # [1, num_of_receptor, 1]

        ish = tf.shape(images) # [batch, h, w, c]

        uspan = castf32(ish[1])
        vspan = castf32(ish[2])

        # UVMap, aka coordinate system
        u = tf.range(start=(-uspan+1)/2,limit=(uspan+1)/2,dtype=tf.float32)
        v = tf.range(start=(-vspan+1)/2,limit=(vspan+1)/2,dtype=tf.float32)
        # U, V -> [hpixels], [wpixels]

        u = tf.expand_dims(u, axis=0)
        u = tf.expand_dims(u, axis=0)

        v = tf.expand_dims(v, axis=0)
        v = tf.expand_dims(v, axis=0)
        # U, V -> [1, 1, hpixels], [1, 1, wpixels]
        # where hpixels = [-0.5...0.5] * image_height
        # where wpixels = [-0.5...0.5] * image_width

        receptor_h = shifted_means[:,:,0:1]
        # [batch, num_of_receptor, 1(h)]
        receptor_w = shifted_means[:,:,1:2]
        # [batch, num_of_receptor, 1(w)]

        # RBF that sum to one over entire x-y plane:
        # integrate
        #   e^(-((x-0.1)^2+(y-0.3)^2)/v) / (v*pi)
        #   dx dy x=-inf to inf, y=-inf to inf, v>0
        # where ((x-0.1)^2+(y-0.3)^2) is the squared distance on the 2D plane

        # UPDATE 20170405: by using SymPy, we got:
        # infitg(exp((-x**2/var + log(1/pi/var)/2)) * exp(-y**2/var +
        # log(1/pi/var)/2)) = 1

        # squared_dist = (smh - u)**2 + (smw - v)**2
        # [batch, num_of_receptor, hpixels, wpixels]

        # density = tf.exp(- squared_dist / variances) / \
        #         (variances * np.pi)
        # [b, n, h, w] / [1, n, 1, 1]
        # should sum to 1

        # optimized on 20170405 and 20170407
        # reduce calculations to a minimum

        oov = 1/variances
        # half_log_one_over_pi_variances = \
        #     - tf.log(variances)*0.5 + (np.log(1/np.pi)* 0.5)
        # log_one_over_pi_var = np.log(1/np.pi) - tf.log(variances)
        # one_over_pi_variances = (1/np.pi) / variances

        # density = tf.exp(\
        #     -(receptor_h-u)**2 / variances + half_log_one_over_pi_variances) * \
        #     tf.exp(\
        #     -(receptor_w-v)**2 / variances + half_log_one_over_pi_variances)

        density_u = tf.exp(\
            -(receptor_h-u)**2 * oov + np.log(1/np.pi)) * oov
        density_v = tf.exp(\
            -(receptor_w-v)**2 * oov) #* sqrt_pi_variances
        # [b, n, h] and [b, n, w]

        # density_u = tf.expand_dims(density_u, axis=3)
        # density_u = tf.expand_dims(density_u, axis=4)
        # density_v = tf.expand_dims(density_v, axis=3)
        # # [b, n, h, 1, 1] and [b, n, w, 1]

        # density = tf.expand_dims(density, axis=4)
        # [b, n, h, w, 1]

        # images = tf.expand_dims(images, axis=1)
        # # [b, h, w, c] -> [b, 1, h, w, c]
        #
        # tmp = images * density_u
        # # [b, 1, h, w, c] * [b, n, h, 1, 1] -> [b, n, h, w, c]
        # tmp = tf.reduce_sum(tmp, axis=[2]) # -> [b, n, w, c]
        # tmp = tmp * density_v # [b, n, w, c] * [b, n, w, 1] -> [b, n, w, c]
        # tmp = tf.reduce_sum(tmp, axis=[2]) # -> [b, n, c]

        # can we transform above into matmul?
        # [b1wc,h] * [bn11,h] -> [bnwc](sum over h)
        # [bnc,w] * [bn1,w] -> [bnc](sum over w)
        # [bnc] -> [b, n, c]
        # tmp = tf.einsum('bhwc,bnh->bnwc',images,density_u)
        # tmp = tf.einsum('bnwc,bnw->bnc',tmp,density_v)

        # turned out Einstein is a genius:
        tmp = tf.einsum('bhwc,bnh,bnw->bnc',images,density_u,density_v)

        # responses = tf.reduce_sum(density * images, axis=[2,3])
        responses = tmp
        # [batch, num_of_receptor, channel]
        return responses

class GRU_Glimpse2D_onepass(Can):
    def __init__(self, num_h, num_receptors, channels, pixel_span=20):
        super().__init__()

        self.channels = channels # explicit
        self.num_h = num_h
        self.num_receptors = num_receptors
        self.pixel_span = pixel_span # how far can the fovea go

        num_in = channels * num_receptors

        self.glimpse2d = g2d = Glimpse2D(num_receptors, pixel_span)
        self.gru_onepass = gop = GRU_onepass(num_in,num_h)
        self.hidden2offset = h2o = Dense(num_h,2)
        # self.glimpse2gru = g2g = Dense(num_in,num_gru_in)

        self.incan([g2d,gop,h2o])

    def __call__(self,i):
        hidden = i[0] # hidden state of gru [batch, dims]
        images = i[1] # input image [NHWC]

        g2d = self.glimpse2d
        # g2g = self.glimpse2gru
        gop = self.gru_onepass
        h2o = self.hidden2offset

        # hidden is of shape [batch, dims], range [-1,1]
        offsets = self.get_offset(hidden) # [batch, 2]

        responses = g2d([images,offsets]) # [batch, num_receptors, channels]
        rsh = tf.shape(responses)
        responses = tf.reshape(responses,shape=(rsh[0],rsh[1]*rsh[2]))

        # responses2 = g2g(responses)
        # responses2 = Act('lrelu')(responses2)
        hidden_new = gop([hidden,responses])
        return hidden_new

    def get_offset(self, hidden):
        # given hidden state of GRU, calculate next step offset
        # hidden is of shape [batch, dims], range [-1,1]
        h2o = self.hidden2offset
        offsets = tf.tanh(h2o(hidden)) # [batch, 2]
        offsets = offsets * self.pixel_span / 2
        return offsets

GRU_Glimpse2D = rnn_gen('GG2D', GRU_Glimpse2D_onepass)

class PermutedConv2D(Can):
    def __init__(self, nip, nop, bottleneck_width, seed, k=3, std=1, *a, **w):
        super().__init__()
        self.nip, self.nop, self.bnw = nip, nop, bottleneck_width
        assert nip >= self.bnw
        assert nop >= nip
        self.num_output_padding = nop - nip

        self.conv = self.add(Conv2D(self.bnw, self.bnw, k=k, std=std, stddev=.5, *a, **w))
        self.seed = seed

        from numpy.random import RandomState
        prng = RandomState(self.seed)

        if self.num_output_padding>0:
            self.io_permtable = prng.randint(self.nip, size = [self.num_output_padding])

        self.offset = prng.randint(self.nip-self.bnw+1)
        self.bo_permtable = prng.randint(self.bnw, size = [self.nop])

    def __call__(self, i):
        # assume [NHWC]
        s = tf.shape(i)

        residual = i

        if self.conv.std>1:
            residual  = AvgPool2D(k=self.conv.k, std=self.conv.std)(residual)

        if self.num_output_padding > 0:
            residual_pad = tf.gather(residual, self.io_permtable, axis=3)
            residual = tf.concat([residual, residual_pad], axis=3)

        piece = i[:,:,:, self.offset: self.offset + self.bnw]
        piece = self.conv(piece)
        piece = Act('lrelu')(piece)

        scattered = tf.gather(piece, self.bo_permtable, axis=3)
        result = scattered + residual*0.707
        return result
