import cv2
import numpy as np

import math
ceil = math.ceil

# input to this function must have shape [N H W C] where C = 1 or 3.
def batch_image_to_array(arr, margin=1, color=None, aspect_ratio=1.1, width=None, height=None):
    import cv2
    num,uh,uw,depth = arr.shape

    patches = num
    if height is None:
        height = max(1,int(np.sqrt(patches)/aspect_ratio))
    if width is None:
        width = ceil(patches/height)

    assert width*height >= patches

    img = np.zeros((height*(uh+margin), width*(uw+margin), 3),dtype=arr.dtype)
    if color is None:
        if arr.dtype == 'uint8':
            img[:,:,1] = 25
        else:
            img[:,:,1] = .1
    else:
        img += color

    index = 0
    for row in range(height):
        for col in range(width):
            if index<num:
                channels = arr[index]
                img[row*(uh+margin):row*(uh+margin)+uh,col*(uw+margin):col*(uw+margin)+uw,:] = channels
            index+=1
    return img

# automatically scale the image up or down, using nearest neighbour.
def autoscale(img,limit=400.):
    # scales = [0.1,0.125,1./6.,0.2,0.25,1./3.,1./2.] + range(100)
    scales = np.hstack([1./np.linspace(10,2,num=9), np.linspace(1,100,num=100)])

    imgscale = limit/float(max(img.shape[0],img.shape[1]))
    for s in reversed(scales):
        if s<=imgscale:
            imgscale=s
            break

    if imgscale!=1.:
        if imgscale<1.:
            # img = cv2.resize(img,dsize=(
            #     int(img.shape[1]*imgscale),int(img.shape[0]*imgscale)),
            #     interpolation=cv2.INTER_LINEAR) # use bilinear
            img = resize_perfect(
                img, img.shape[0]*imgscale, img.shape[1]*imgscale,
                a = 1,
            )
        else:
            img = cv2.resize(img,dsize=(
                int(img.shape[1]*imgscale),int(img.shape[0]*imgscale)),
                interpolation=cv2.INTER_NEAREST) # use nn

    return img,imgscale

def resize_of_interpolation(interpolation):
    def resize_interpolation(img, h, w):
        return cv2.resize(img,
            dsize=(int(w), int(h)),
            interpolation=interpolation)
    return resize_interpolation

resize_linear = resize_of_interpolation(cv2.INTER_LINEAR)
resize_cubic = resize_of_interpolation(cv2.INTER_CUBIC)
resize_lanczos = resize_of_interpolation(cv2.INTER_LANCZOS4)
resize_nearest = resize_of_interpolation(cv2.INTER_NEAREST)
resize_area = resize_of_interpolation(cv2.INTER_AREA)

def lanczos_kernel(radius, a=2):
    halfskirt = ceil(radius*a-1)
    fullskirt = halfskirt*2 + 1
    scaled_hs = halfskirt/radius

    y = np.linspace(-scaled_hs, scaled_hs, fullskirt)

    lanczos = np.where((-a<y) * (y<a), np.sinc(y)*np.sinc(y/a), 0)
    return (lanczos / lanczos.sum()).astype('float32')

def lanczos_filter(img, yradius, xradius, a=2):
    lanczosy = lanczos_kernel(yradius, a=a)
    lanczosx = lanczos_kernel(xradius, a=a)
    lanczosy.shape = lanczosy.shape+(1,)
    lanczosx.shape = (1,) + lanczosx.shape
    img = cv2.filter2D(img, -1, lanczosy)
    img = cv2.filter2D(img, -1, lanczosx)
    return img

# prefilter then resize.
def resize_perfect(img, h, w, cubic=False, a=2):
    assert 1<=a and a<5

    hr = img.shape[0]/h
    wr = img.shape[1]/w

    if hr > 1 or wr > 1:
        img = lanczos_filter(img, hr, wr, a=a)
        # lanczosy = lanczos_kernel(hr, a=a)
        # lanczosx = lanczos_kernel(wr, a=a)
        #
        # lanczosy.shape = lanczosy.shape+(1,)
        # lanczosx.shape = (1,) + lanczosx.shape
        #
        # img = cv2.filter2D(img, -1, lanczosy)
        # img = cv2.filter2D(img, -1, lanczosx)
    else:
        pass

    if cubic:
        return resize_cubic(img, h, w)
    else:
        return resize_linear(img, h, w)

resize_autosmooth = resize_perfect

# use resize_perfect() instead for downsampling.
def resize_box(img, h, w):
    # check scale
    scale = h / img.shape[0]

    if scale > 1:
        return resize_linear(img, h, w)
    else:
        return resize_area(img, h, w)

def show_autoscaled(img,limit=400.,name=''):
    im,ims = autoscale(img,limit=limit)
    cv2.imshow(name+str(img.shape)+' gened scale:'+str(ims),im)
    cv2.waitKey(1)

def show_batch_autoscaled(arr,limit=400.,name=''):
    img = batch_image_to_array(arr)
    show_autoscaled(img,limit,name)

# import matplotlib.pyplot as plt
# class plotter:
#     def __init__(self):
#         plt.ion()
#
#         self.x = []
#         self.y = []
#         self.fig = plt.figure()
#         self.ax = self.fig.add_subplot(1,1,1)
#
#     def pushy(self,y):
#         self.y.append(y)
#         if len(self.x)>0:
#             self.x.append(self.x[-1]+1)
#         else:
#             self.x.append(0)
#     def newpoint(self,y):
#         self.pushy(y)
#
#     def show(self):
#         self.ax.clear()
#         self.ax.plot(self.x,self.y)
#         plt.draw()

if __name__ == '__main__':
    lanczos = prefilter_lanczos_kernel(.1,.5)
    print('sum', lanczos.sum())
    print(lanczos, lanczos.shape)
    show_autoscaled(lanczos*0.5+0.5,limit=50)
    cv2.waitKey(0)