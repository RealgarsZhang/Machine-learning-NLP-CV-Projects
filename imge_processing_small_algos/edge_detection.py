import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# load the image
I = np.asarray(Image.open('edge_detection_image.jpg').convert("L"), dtype=float)

# print the size of loaded image
print("size of loaded image is ", I.shape)

# show the loaded image
plt.imshow(I, cmap="gray")

# sobel edge detector
# https://en.wikipedia.org/wiki/Sobel_operator

Gy = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
Gx = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
m, n = Gx.shape
edge_x_img = np.zeros_like(I)
edge_y_img = np.zeros_like(I)
edge_img = np.zeros_like(I)

for i in range(I.shape[0] - m//2 * 2):
    for j in range(I.shape[1] - n//2 *2):
        edge_x_img[i+m//2,j+n//2] = sum(sum(np.multiply(I[i:i+m, j:j+n], Gx)))
        edge_y_img[i+m//2,j+n//2] = sum(sum(np.multiply(I[i:i+m, j:j+n], Gy)))
        edge_img[i+m//2, j+n//2] = np.sqrt(edge_x_img[i+m//2,j+n//2] ** 2 + edge_y_img[i+m//2,j+n//2] ** 2)

plt.imshow(edge_img, cmap="gray")

# implement edge detection by laplacian operator
Gy = np.array([[0,1,0],[0,-2,0],[0,1,0]])
Gx = np.array([[0,0,0],[1,-2,1],[0,0,0]])
m, n = Gx.shape
edge_x_img = np.zeros_like(I)
edge_y_img = np.zeros_like(I)
edge_img = np.zeros_like(I)

for i in range(I.shape[0] - m//2 * 2):
    for j in range(I.shape[1] - n//2 *2):
        edge_x_img[i+m//2,j+n//2] = sum(sum(np.multiply(I[i:i+m, j:j+n], Gx)))
        edge_y_img[i+m//2,j+n//2] = sum(sum(np.multiply(I[i:i+m, j:j+n], Gy)))
        edge_img[i+m//2, j+n//2] = edge_x_img[i+m//2,j+n//2]+edge_y_img[i+m//2,j+n//2]

plt.imshow(edge_img, cmap="gray")
