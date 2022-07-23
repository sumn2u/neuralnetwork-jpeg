import math
import numpy as np
import cv2

# generate a kernel for motion blur.
# dim is kernel size
# angle is kernel angle

def generate_motion_blur_kernel(dim=3,angle=0.,threshold_factor=1.0,divide_by_dim=True,vector=None):
    if vector is None: #use angle as input
        radian = angle/360*math.pi*2 + math.pi/2

        if dim<2:
            return None

        # first, generate xslope and yslope
        offset = (dim-1)/2 # zero-centering
        gby,gbx = np.mgrid[0-offset:dim-offset:1.,0-offset:dim-offset:1.] # assume dim=3, 0:dim -> -1,0,1

        # then mix the slopes according to angle
        gbmix = gbx * math.cos(radian) - gby * math.sin(radian)

    else: #use vector as input
        y,x = vector
        l2 = math.sqrt(x**2+y**2)+1e-2
        ny,nx = y/l2,x/l2

        dim = int(max(abs(x),abs(y)))
        if dim<2:
            return None

        # first, generate xslope and yslope
        offset = (dim-1)/2 # zero-centering
        gby,gbx = np.mgrid[0-offset:dim-offset:1.,0-offset:dim-offset:1.] # assume dim=3, 0:dim -> -1,0,1

        gbmix = gbx * ny - gby * nx

    # threshold it into a line
    kernel = (threshold_factor - gbmix*gbmix).clip(min=0.,max=1.)

    if divide_by_dim: # such that the brightness wont change
        kernel /= np.sum(kernel)
    else:
        pass

    return kernel.astype('float32')

# apply motion blur to the image.
def apply_motion_blur(im,dim,angle):
    kern = generate_motion_blur_kernel(dim,angle)
    if kern is None:
        return im
    imd = cv2.filter2D(im,cv2.CV_32F,kern)
    return imd

def apply_vector_motion_blur(im,vector):
    kern = generate_motion_blur_kernel(vector=vector)
    if kern is None: # if kernel too small to be generated
        return im
    imd = cv2.filter2D(im,cv2.CV_32F,kern)
    return imd

# apply rotation and scaling to an image, while keeping it in the center of the resulting square image.
# img should be an image of shape [HWC]
def rotate_scale(img,angle=0.,scale=1.):
    ih,iw = img.shape[0:2]

    side = int(max(ih,iw) * scale * 1.414) # output image height and width
    center = side//2

    # 1. scale
    orig_points = np.array([[iw/2,0],[0,ih/2],[iw,ih/2]],dtype='float32')
    # x,y coord of top left right corner of input image.

    translated = np.array([[center,center-ih*scale/2],[center-iw*scale/2,center],[center+iw*scale/2,center]],dtype='float32')
    # get affine transform matrix
    at = cv2.getAffineTransform(orig_points,translated)
    at = np.vstack([at,[0,0,1.]])

    # 2. rotate
    rm = cv2.getRotationMatrix2D((center,center),angle,1)
    # per document:
    # coord - rotation center
    # angle – Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
    rm = np.vstack([rm,[0,0,1.]])

    # 3. combine 2 affine transform
    cb = np.dot(rm,at)
    # print(cb)

    # 4. do the transform
    res = cv2.warpAffine(img,cb[0:2,:],(side,side),flags=cv2.INTER_CUBIC)
    return res

# calculate intersection of two integer roi. returns the intersection.
# assume (y,x) or (h,w)
def intersect(topleft1,topleft2,size1,size2):
    x_topleft = max(topleft1[1],topleft2[1])
    y_topleft = max(topleft1[0],topleft2[0])
    x_bottomright = min(topleft1[1]+size1[1], topleft2[1]+size2[1])
    y_bottomright = min(topleft1[0]+size1[0], topleft2[0]+size2[0])

    if x_topleft<x_bottomright and y_topleft<y_bottomright:
        return [y_topleft,x_topleft], [y_bottomright,x_bottomright]
        # [y_bottomright-y_topleft,x_bottomright-x_topleft]
    else:
        return None

# same as above, but return roi-ed numpy array views
def intersect_get_roi(bg,fg,offset=[0,0],verbose=True, return_numbers=False):
    bgh,bgw = bg.shape[0:2]
    fgh,fgw = fg.shape[0:2]

    # obtain roi in background coords
    isect = intersect([0,0], offset, [bgh,bgw], [fgh,fgw])
    if isect is None:
        if verbose==True:
            print('(intersect_get_roi)warning: two roi of shape',bg.shape,fg.shape,'has no intersection.')
        return None

    tl,br = isect
    # print(isect)
    bgroi = bg[tl[0]:br[0], tl[1]:br[1]]
    bgroi_numbers = [tl[0], br[0], tl[1], br[1]]

    # obtain roi in fg coords
    tl,br = isect = intersect([-offset[0],-offset[1]],[0,0],[bgh,bgw],[fgh,fgw])
    # print(isect)
    fgroi = fg[tl[0]:br[0], tl[1]:br[1]]
    fgroi_numbers = [tl[0], br[0], tl[1], br[1]]

    if return_numbers:
        return fgroi,bgroi, fgroi_numbers, bgroi_numbers
    else:
        return fgroi,bgroi

# same as above, but deal only with the dimensions
def intersect_get_roi_numbers(bgshape,fgshape,offset=[0,0],verbose=True):
    bgh,bgw = bgshape[0:2]
    fgh,fgw = fgshape[0:2]

    # obtain roi in background coords
    isect = intersect([0,0], offset, [bgh,bgw], [fgh,fgw])
    if isect is None:
        if verbose==True:
            print('(intersect_get_roi)warning: two roi of shape ({} {}) ({} {}) has no intersection.'.format(bgh,bgw,fgh,fgw))
        return None
    else:
        tl, br = isect
        bgroi_numbers = [tl[0], br[0], tl[1], br[1]]

        tl, br = isect = intersect([-offset[0],-offset[1]],[0,0],[bgh,bgw],[fgh,fgw])
        fgroi_numbers = [tl[0], br[0], tl[1], br[1]]

        return bgroi_numbers, fgroi_numbers

# place one image atop another, no alpha involved
def direct_composite(bg, fg, offset=[0,0]):
    isectgr = intersect_get_roi(bg,fg,offset,verbose=None)
    if isectgr is None:
        return bg
    else:
        fgroi,bgroi = isectgr

    bgroi[:] = fgroi[:]
    return bg

# alpha composition.
# bg shape: [HW3] fg shape: [HW4] dtype: float32
def alpha_composite(bg,fg,offset=[0,0],verbose=True):
    isectgr = intersect_get_roi(bg,fg,offset,verbose=verbose)
    if isectgr is None:
        return bg
    else:
        fgroi,bgroi = isectgr

    # print('alphacomp',bgroi.shape,fgroi.shape)
    alpha = fgroi[:,:,3:3+1]
    bgroi[:] = bgroi[:] * (1 - alpha) + fgroi[:,:,0:3] * alpha

    return bg

# same as above but
# takes separated alpha channel as input (size == foreground)
# allow any number of channels as input
def alpha_composite_separated(bg, fg, alpha, offset=[0,0], verbose=True):
    # assert bg.shape[2] == fg.shape[2]
    assert fg.shape[0] == alpha.shape[0]
    assert fg.shape[1] == alpha.shape[1]

    if len(alpha.shape)==2: alpha.shape+=(1,)

    isectgr = intersect_get_roi(bg,fg,offset,verbose=verbose, return_numbers=True)
    if isectgr is None:
        return bg
    else:
        fgroi, bgroi, fgroi_numbers, bgroi_numbers = isectgr

    fn = fgroi_numbers
    alpha = alpha[fn[0]:fn[1], fn[2]:fn[3]]

    bgroi[:] = bgroi[:] * (1-alpha) + fgroi[:] * alpha
    return bg

# offer a full set of settings for alpha composition.
# https://en.wikipedia.org/wiki/Alpha_compositing
# IMPORTANT: the input images are expected to be of PREMULTIPLIED alpha
# to prevent numerical problems blending small values of alpha.
def alpha_composite_full(bg, fg, bgalpha, fgalpha, offset=[0,0], verbose=True):
    if bg.ndim==2: bg.shape+=(1,)
    if fg.ndim==2: fg.shape+=(1,)
    if bgalpha.ndim==2: bgalpha.shape+=(1,)
    if fgalpha.ndim==2: fgalpha.shape+=(1,)
    assert bg.ndim==3 and fg.ndim==3 and fgalpha.ndim==3 and bgalpha.ndim==3
    assert fg.shape[0] == fgalpha.shape[0] and fg.shape[1] == fgalpha.shape[1]
    assert bg.shape[0] == bgalpha.shape[0] and bg.shape[1] == bgalpha.shape[1]
    assert fgalpha.shape[2] == 1
    assert bgalpha.shape[2] == 1

    isectgr = intersect_get_roi_numbers(bg.shape,fg.shape,offset,verbose=verbose)
    if isectgr is None:
        return bg, bgalpha
    else:
        bgroi_numbers, fgroi_numbers = isectgr

    fn = fgroi_numbers
    bn = bgroi_numbers

    fgroi = fg[fn[0]:fn[1], fn[2]:fn[3]]
    fgalpharoi = fgalpha[fn[0]:fn[1], fn[2]:fn[3]]
    bgroi = bg[bn[0]:bn[1], bn[2]:bn[3]]
    bgalpharoi = bgalpha[bn[0]:bn[1], bn[2]:bn[3]]

    # color mixing
    negfgalpha = 1 - fgalpharoi
    bgalpharoi[:] = fgalpharoi + bgalpharoi * negfgalpha
    bgroi[:] = fgroi + bgroi * negfgalpha
    return bg, bgalpha

# same as above but assume both input is of [HW4].
def alpha_composite_full_combined(bg, fg, offset=[0,0], verbose=True):
    bg, bgalpha = alpha_composite_full(bg[:,:,0:3], fg[:,:,0:3], bg[:,:,3:4], fg[:,:,3:4], offset, verbose)
    return bg

# pixel displacement
def displace(img,dy,dx):
    assert img.shape[0:2] == dy.shape[0:2]
    assert img.shape[0:2] == dx.shape[0:2]

    ih,iw = img.shape[0:2]
    rowimg,colimg = np.mgrid[0:ih,0:iw]

    rowimg += dy.astype('int32')
    colimg += dx.astype('int32')

    rowimg = np.clip(rowimg,a_max=ih-1,a_min=0)
    colimg = np.clip(colimg,a_max=iw-1,a_min=0)

    res = img[rowimg,colimg]
    return res

if __name__ == '__main__':
    def test(combined=False):
        if combined == False:
            black3 = np.zeros((256,256,3)).astype('float32')
            black1 = np.zeros((256,256,1)).astype('float32')
            red = black3.copy()
            red[:,:,2]+=1

            green = black3.copy()
            green[:,:,1]+=1

            blue = black3.copy()
            blue[:,:,0]=1

            mask_b = black1.copy()+.5

            mask_g = black1.copy()
            mask_g[50:150, 50:150] = .5

            mask_r = black1.copy()
            mask_r[100:200, 100:200] = .5

            premult_r = red*mask_r
            premult_g = green*mask_g

            bg = black3.copy()
            mask_bg = black1.copy()

            alpha_composite_full(bg, premult_r, mask_bg, mask_r)
            cv2.imshow('red over trans', bg)

            alpha_composite_full(bg, premult_g, mask_bg, mask_g)

            cv2.imshow('green over red over trans', bg)
            cv2.imshow('alpha: green over red over trans', mask_bg)

            alpha_composite_full(bg, blue, mask_bg, mask_b, [130,130])
            cv2.imshow('b o g o r o t', bg)
            cv2.imshow('alpha: b o g o r o t', mask_bg)

            w,wm = alpha_composite_full(bg*0+1, bg, mask_bg*0+1, mask_bg)
            cv2.imshow('white composite', w)
        else:
            black4 = np.zeros((256,256,4)).astype('float32')
            red = black4.copy()
            green = black4.copy()
            blu = black4.copy()

            def prem(k):
                k[:,:,0:3] *= k[:,:,3:4]

            red[:,:,2] +=1
            red[100:200, 100:200, 3] = .5
            prem(red)

            green[:,:,1] += 1
            green[50:150, 50:150, 3] = .5
            prem(green)

            bg = black4.copy()
            white = black4.copy()+1

            alpha_composite_full_combined(bg, red)
            alpha_composite_full_combined(bg, green)

            cv2.imshow('bg',bg)
            alpha_composite_full_combined(white, bg)
            cv2.imshow('white',white)

        cv2.waitKey(0)

    test(combined=False)
    test(combined=True)