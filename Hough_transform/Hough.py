import numpy as np
import matplotlib.pyplot as plt
import cv2
%matplotlib inline

input_img_names = ["input_0.png", "input_1.png", "input_2.png"]
input_imgs = [None, None, None]

for idx, name in enumerate(input_img_names):
    input_imgs[idx] = cv2.imread(name)

def detect_edges(input_img):
    '''
        Given an input image, output an edge-detected image.
        '''
    return cv2.Canny(input_img,65,80)

def generate_hough_accumulator(img, theta_num_bins, d_num_bins):
    '''
        Given an input image, and # bins for theta and d, return
        a (d_num_bins, theta_num_bins) accumulator matrix for Hough space.
        '''
    m,n = img.shape
    d_min = -m-n
    d_max = m+n
    d_bin_size = (d_max-d_min)/d_num_bins
    pi = 3.1415926
    theta_bin_size = 2*pi/theta_num_bins
    theta = np.array(range(theta_num_bins))*theta_bin_size
    Hough_matrix = np.zeros((theta_num_bins,d_num_bins))
    for i in range(m):
        for j in range(n):
            if img[i,j] == 255:
                idx = np.floor((i*np.sin(theta)+j*np.cos(theta)-d_min)/d_bin_size).astype("int")
                Hough_matrix[range(theta_num_bins),idx] += 1
    
    return Hough_matrix/np.max(Hough_matrix)*255

def line_finder(orig_img, hough_img, hough_threshold):
    '''
        Given the original img, hough image (the accumulator matrix
        from prev part), and hough threshold, return the input image with
        detected lines drawn on it.
        '''
    
    pi = 3.1415926
    epsilon = 0.0001
    m = orig_img.shape[0]
    n = orig_img.shape[1]
    #print (m,n)
    line_img = orig_img.copy()
    #print (line_img.shape)
    d_min = -m-n
    d_max = m+n
    theta_num_bins = hough_img.shape[0]
    d_num_bins = hough_img.shape[1]
    #print (hough_img.shape)
    points = np.argwhere(hough_img>hough_threshold)
    #print (points)
    for i in range(points.shape[0]):
        theta = points[i,0]*2*pi/theta_num_bins
        d = points[i,1]/d_num_bins*(d_max-d_min)+d_min
        #print (points[i,:],theta,d)
        if abs(np.cos(theta))<epsilon:
            x_coor  = int(d/np.sin(theta))
            pt1 = (x_coor,0)
            pt2 = (x_coor,n)
            cv2.line(line_img,pt1[::-1],pt2[::-1],(255,255,255),2)# row col reversed
        else:
            y1 = int(d/np.cos(theta))
            y2 = int((d-m*np.sin(theta))/np.cos(theta))
            pt1 = (0,y1)
            pt2 = (m,y2)
            cv2.line(line_img,pt1[::-1],pt2[::-1],(255,255,255),2)
    #cv2.line(line_img,(0,0),(n,m),(255,255,255),2)
#print (line_img.shape)
    return line_img,hough_img

theta_num_bins =  [300, 300, 300]
d_num_bins =  [1000, 1000, 800]
hough_thresholds =  [100, 100, 100]


edge_imgs = [None, None, None]

for idx, input_img in enumerate(input_imgs):
    edge_imgs[idx] = detect_edges(input_img)
    
    plt.imshow(edge_imgs[idx], cmap = 'gray'); plt.show()



hough_accumulator_imgs = [None, None, None]

for idx in range(len(input_imgs)):
    img = edge_imgs[idx]
    hough_accumulator = generate_hough_accumulator(img, theta_num_bins[idx], d_num_bins[idx])
    
    # We'd like to save the hough accumulator array as an image to
    # visualize it. The values should be between 0 and 255 and the
    # data type should be uint8.
    hough_accumulator_imgs[idx] = hough_accumulator.astype("uint8")
    #print (np.max(hough_accumulator_imgs[idx]))
    print(np.argwhere(hough_accumulator_imgs[idx]>230))
    plt.imshow(hough_accumulator_imgs[idx], cmap = "gray"); plt.show()

line_imgs = [None, None, None]

for idx in range(len(input_imgs)):
    orig_img = input_imgs[idx]
    hough_img = hough_accumulator_imgs[idx]
    
    line_img, hough_img = line_finder(orig_img, hough_img, hough_thresholds[idx])
    
    # The values of line_img should be between 0 and 255 and the
    # data type should be uint8.
    #
    # Here we cast line_img to uint8 if you have not done so, otherwise
    # imwrite will treat line_img as a double image and save it to an
    # incorrect result.
    #print (line_img.shape)
    line_imgs[idx] = line_img.astype("uint8")
    #print (line_img.shape)
    #print (orig_img.shape)
    #cp = orig_img.copy()
    #cv2.line(cp,(0,0),(400,600),(255,255,255),5)
    #cv2.line(line_imgs[idx],(0,0),(400,600),(255,255,255),5)
    #print (line_imgs[idx].shape)
    plt.imshow(line_imgs[idx]); plt.show()
#plt.imshow(cp); plt.show()

