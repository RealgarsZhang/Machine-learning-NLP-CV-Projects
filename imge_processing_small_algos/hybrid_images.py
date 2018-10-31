import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# load the image
dog = np.asarray(Image.open('dog.jpg').convert("L"), dtype=float) / 255
dog.setflags(write=1)

# load the image
koala = np.asarray(Image.open('koala.jpg').convert("L"), dtype=float) / 255
koala.setflags(write=1)

f, axarr = plt.subplots(1,2)
axarr[0].imshow(dog, cmap='gray')
axarr[1].imshow(koala, cmap='gray')


#get high and low frequency part
from scipy.fftpack import fft2,ifft2
import pylab

koala_fft = np.fft.fftshift(np.fft.fft2(koala))

koala_high_freq_fft = koala_fft.copy()
m,n = koala_fft.shape
koala_high_freq_fft[m//2-10:m//2+10,n//2-10:n//2+10] = 0+0.j
koala_high_freq = np.absolute(np.fft.ifft2((koala_high_freq_fft)))


dog_fft = np.fft.fftshift(np.fft.fft2(dog))
dog_low_freq_fft = np.zeros(dog_fft.shape)*(0+0.j)
dog_low_freq_fft[m//2-10:m//2+10,n//2-10:n//2+10] = dog_fft[m//2-10:m//2+10,n//2-10:n//2+10]
dog_low_freq = np.absolute(ifft2((dog_low_freq_fft)))


f, axarr = plt.subplots(1,2)
axarr[0].imshow(koala_high_freq, cmap='gray')
axarr[1].imshow(dog_low_freq, cmap='gray')

# get the hybrid image and sampling results
import scipy.misc
from skimage.transform import resize

I = koala_high_freq+dog_low_freq
plt.imshow(koala_high_freq+dog_low_freq,cmap = 'gray')

f, axarr = plt.subplots(1,5)
temp = I
for i in range(5):
    temp = scipy.misc.imresize(temp, 0.5)
    axarr[i].imshow(temp,cmap='gray')


