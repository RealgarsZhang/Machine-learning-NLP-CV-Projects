import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# load the image
I = np.asarray(Image.open('pug_image.jpg'))
I.setflags(write=1)

# print the size of loaded image
print("size of loaded image is ", I.shape)

# show the loaded image
plt.imshow(I)

# todo:
# recover the original image
# use the third channel as benchmark

def SSD(I1,I2):#This is actually the square root of SSD
    return np.linalg.norm(I1-I2,'fro')

plt.imshow(I[:,:,0])
benchmark = I[:,:,2].astype('float')
res = I.copy()
#fix 0th channel
shift_0 = 0
shifted_result_0 = I[:,:,0].astype('float')
cur_ssd = SSD(shifted_result_0,benchmark)
for i in range(1,45): # assume shift less than 45
    shifted_image = np.roll(I[:,:,0],i,axis = 0)
    temp_ssd = SSD(shifted_image,benchmark)
    if temp_ssd<cur_ssd:
        cur_ssd = temp_ssd
        shift_0 = i
        shifted_result_0 = shifted_image
# fix 1st channel
shift_1 = 0
shifted_result_1 = I[:,:,1].astype('float')
cur_ssd = SSD(shifted_result_1,benchmark)

for i in range(1,45):
    shifted_image = np.roll(I[:,:,1],-i,axis = 1)
    temp_ssd = SSD(shifted_image,benchmark)
    if temp_ssd<cur_ssd:
        cur_ssd = temp_ssd
        shift_1 = -i
        shifted_result_1 = shifted_image

res[:,:,0] = shifted_result_0
res[:,:,1] = shifted_result_1
res[:,:,2] = benchmark
plt.imshow(res)
