import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# load the image
I = np.asarray(Image.open('noisy_image.jpg').convert("L"), dtype=float)

# print the size of loaded image
print("size of loaded image is ", I.shape)

# show the loaded image
plt.imshow(I, cmap="gray")

#mean filter
my_filter = np.ones([5,5])
m, n = my_filter.shape
my_filter = my_filter / m / n
new_img = np.zeros_like(I)

for i in range(I.shape[0] - m//2 * 2):
    for j in range(I.shape[1] - n//2 * 2):
        new_img[i+m//2,j+n//2] = sum(sum(np.multiply(I[i:i+m, j:j+n], my_filter)))

plt.imshow(new_img, cmap="gray")

#Gaussian filter
import math
pi = 3.1415
sigma = 1.2
my_filter = np.ones([5,5])
m, n = my_filter.shape
for i in range(m):
    for j in range(n):
        radius = (i-m//2)**2 + (j-n//2)**2
        my_filter[i,j] = 1/2/pi/sigma/sigma*math.exp(-radius/2/sigma/sigma)
#my_filter = my_filter / m / n
#print (my_filter)
new_img = np.zeros_like(I)

for i in range(I.shape[0] - m//2 * 2):
    for j in range(I.shape[1] - n//2 * 2):
        new_img[i+m//2,j+n//2] = sum(sum(np.multiply(I[i:i+m, j:j+n], my_filter)))

plt.imshow(new_img, cmap="gray")
