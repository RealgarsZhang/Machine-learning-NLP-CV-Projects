import numpy as np
import math
from scipy import signal

# modeled on https://www.cs.ubc.ca/~little/cpsc425/hw6.html

def rel_error(x, y):
  """ returns relative error """
  # code is adapted from: http://cs231n.stanford.edu/
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def gaussian2D(sigma=0.5):
    """
    2D gaussian filter
    """

    size = int(math.ceil(sigma * 6))
    if (size % 2 == 0):
        size += 1
    r, c = np.ogrid[-size / 2: size / 2 + 1, -size / 2: size / 2 + 1]
    g = np.exp(-(c * c + r * r) / (2. * sigma ** 2))
    g = g / (g.sum() + 0.000001)
    
    return g

def box2D(n):
    """
    2D box filter
    """
    
    box = np.full((n, n), 1. / (n * n))

    return box


def calculate_derivatives(i1, i2, sigma=0.5, n=3):
    """
    Derive Ix, Iy and It in this function

    To derive the spatial derivative in one image, you need to smooth the image with gaussian filter,
    and calculate the derivative, signal.convolve2d and np.gradient might be useful here
    """
    g = gaussian2D(sigma)
    i1_smoothed = signal.convolve2d(i1,g,mode="same")
    #print(g.shape)
    i2_smoothed = signal.convolve2d(i2,g,mode="same")
    ix, iy = np.gradient(i1_smoothed)
    """
    To derive the temporal derivative in two images, you need to filter the images with box filters,
    and then calculate the difference between the results
    """
    box = box2D(n)
    i1_boxed = signal.convolve2d(i1,box,mode="same")
    i2_boxed = signal.convolve2d(i2,box,mode="same")
    it = i2_boxed - i1_boxed
    return ix,iy,it

def optical_flow(i1, i2, x, y, window, sigma=0.5, n=3):
    """
    use calculate_derivatives to obtain Ix, Iy and It, then use the window size to crop the derivatives around the image
    location x, y. To calculate the pseudo inverse, you can use the pinv function included in numpy
    :param i1: the first frame
    :param i2: the second frame
    :param x: location to calculate optical flow
    :param y: location to calculate optical flow
    :param window: size of the window
    :param sigma: smoothing coefficient
    :param n: box filter size
    :return: u, v
    """
    ix,iy,it = calculate_derivatives(i1,i2,sigma,n)
    window_list = []
    for i in range(x-n//2,x+n//2+1):
        for j in range(y-n//2,y+n//2+1):
            if 0<=i<i1.shape[0] and 0<=j<i1.shape[1]:
                window_list.append([i,j])
    idx0 = list(map(lambda t: t[0], window_list))
    idx1 = list(map(lambda t: t[1], window_list))
    c1 =  ix[idx0,idx1].reshape((len(window_list),1))
    c2 =  iy[idx0,idx1].reshape((len(window_list),1))
    A = np.hstack((c1,c2))
    b = -it[idx0,idx1].reshape((len(window_list),1))
    inv_A = np.linalg.pinv(A)
    return inv_A.dot(b)

# test the calculate_derivatives function
a = np.arange(50, step=2).reshape((5,5))
b = np.roll(a, 1, axis=1)
ix, iy, it = calculate_derivatives(a, b, 3, 3)
#print(ix.shape)
#print(iy.shape)
correct_ix = np.array([[ 1.19566094,  1.44638748,  1.60119287,  1.62253849,  1.50447539],
                        [ 1.0953402,   1.32258973,  1.4614055,   1.4781469,   1.36814814],
                        [ 0.7722809,   0.92753122,  1.01928721,  1.02535579,  0.94404567],
                        [ 0.25022598,  0.29355951,  0.31471869,  0.30865011,  0.27704506],
                        [-0.04909038, -0.06915144, -0.0875189,  -0.09965607, -0.10218035]])
correct_iy = np.array([[ 0.81434768,  0.67021012,  0.31799052, -0.11348914, -0.33688675],
                       [ 1.06507422,  0.87297609,  0.40606602, -0.16184788, -0.45494985],
                       [ 1.26884674,  1.03627543,  0.47354769, -0.2067465,  -0.55688427],
                       [ 1.37557486,  1.1199824,   0.5038906,  -0.23708941, -0.61757009],
                       [ 1.3555138,   1.10076814,  0.48863828, -0.24442014, -0.62009437]])
correct_it = np.array([[ 1.33333333,  0.88888889, -1.33333333, -1.33333333, -0.88888889],
                       [ 2.,          1.33333333, -2.,         -2.,         -1.33333333],
                       [ 2.,          1.33333333, -2.,         -2.,         -1.33333333],
                       [ 2.,          1.33333333, -2.,         -2.,         -1.33333333],
                       [ 1.33333333,  0.88888889, -1.33333333, -1.33333333, -0.88888889]])

print('Testing derivatives:')
print('Ix difference: ', rel_error(ix, correct_ix))
print('Iy difference: ', rel_error(iy, correct_iy))
print('It difference: ', rel_error(it, correct_it))

u, v = optical_flow(a, b, 2, 2, 3, 3, 3)
#correct_u = -3.029245564049842
#correct_v = 1.5425850321489603
correct_v =-1.84228885348387
correct_u = 1.4660487320722453
print('Testing derivatives:')
print('u difference: ', rel_error(u, correct_u))
print('v difference: ', rel_error(v, correct_v))
