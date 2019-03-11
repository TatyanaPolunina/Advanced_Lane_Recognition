import numpy as np
import cv2
import matplotlib.pyplot as plt

def filter_by_threshold(binary, threshold):
    sxbinary = np.zeros_like(binary)
    sxbinary[(binary >= threshold[0]) & (binary <= threshold[1])] = 1
    return sxbinary

def get_sobel_binary(image,  threshold, orient='x'):
    # Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sobel x
    if (orient == 'x'):
        is_x = 1; is_y = 0
    else:
        is_x = 0; is_y = 1
    sobelx = cv2.Sobel(gray, cv2.CV_64F, is_x, is_y)
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))   
    return filter_by_threshold(scaled_sobel, threshold)

def get_hls_s_binary(image, threshold):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    return filter_by_threshold(s_channel, threshold)


def get_dir_binary(image, sobel_kernel=3, thresh=(0, np.pi/2)):    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    arctan_sobel = np.arctan2(abs_sobely, abs_sobelx)   
    return filter_by_threshold(arctan_sobel, thresh);

def get_mag_binary(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    magn = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*magn/np.max(magn))    
    return filter_by_threshold(arctan_sobel, thresh)

def to_binary_image(binary):
    mask_color = 255;
    mask_img = np.full((binary.shape[0],binary.shape[1]), 0)
    mask_img[np.where(binary == 1)] = mask_color;    
    return mask_img;
