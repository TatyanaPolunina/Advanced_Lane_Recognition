import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageWarper:
    transform_Matrix = []
    def __init__(self, src_polygon, dst_polygon):
        self.transform_Matrix = cv2.getPerspectiveTransform(src_polygon, dst_polygon)
        
    def warp_image(self, img):
        img_size = (img.shape[1], img.shape[0]);
        warp_img = cv2.warpPerspective(img, self.transform_Matrix, img_size, flags=cv2.INTER_LINEAR)
        plt.imshow(warp_img)
        return warp_img
        
        