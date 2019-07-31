import lane_detection as ld

import matplotlib.pyplot as plt
import numpy as np
import cv2
import lines
import calibration as cb;
import binary_outputs as bo
import lane_detection as ld
import transformation as tr

def get_binary(img):
    hls_s_binary = bo.get_hls_s_binary(img, (210, 255));
    scale_x_binary = bo.get_sobel_binary(img, (70, 100));
    yellow_binary = bo.get_yellow_binary(img, (220, 255));
    mag_binary = bo.get_mag_binary(img, 5, (70, 220))
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(hls_s_binary)
    combined_binary[(hls_s_binary == 1) | (scale_x_binary == 1) | (yellow_binary == 1) | (mag_binary == 1)] = 1
    return combined_binary;

def get_model_polygon (img):
    img_size = [img.shape[1], img.shape[0]]
    src = np.float32(
     [[(img_size[0] * 11/ 24), img_size[1] * 5/8],
     [((img_size[0] / 8) ), img_size[1]],
     [(img_size[0] *  11 / 12) , img_size[1]],
     [(img_size[0] * 13/ 24 ), img_size[1] * 5 / 8]])
    dst = np.float32(
     [[(img_size[0] / 6-50), 0],
     [(img_size[0] / 6-50), img_size[1]],
     [(img_size[0] * 5 / 6 -50), img_size[1]],
     [(img_size[0] * 5 / 6-50), 0]])
    return src, dst


class ImageProcessor:
    camera_calibrator = []
    wrap_polygon = []
    warper = []
    inverted_warper = []
    lane_detector = []
    def __init__(self, camera_calibrator):
        self.camera_calibrator = camera_calibrator
        self.lane_detector = ld.LaneDetector()

    def init_warpers(self, image):
        src, dst = get_model_polygon(image);
        self.warper = tr.ImageWarper( src, dst);
        self.inverted_warper = tr.ImageWarper( dst, src)
        
    def process(self, image, need_to_print_position = False):
        undist = self.camera_calibrator.undist_image(image)
        binary = get_binary(undist)
        if not self.warper:
            self.init_warpers(image)
            
        warp_img = self.warper.warp_image(binary)
        # View your output
        self.lane_detector.find_next_poly(warp_img);
        if need_to_print_position:
            print("Curvature radius ", self.lane_detector.get_curvature(warp_img))
            print("Offset from center ", self.lane_detector.get_offset(warp_img), " meters")
        warp_img = warper.warp_image(binary)
        # View your output
        self.lane_detector.find_next_poly(warp_img);
        if need_to_print_position:
            print("Curvature ", self.lane_detector.get_curvature(warp_img))
            print("Center offset ", self.lane_detector.get_offset(warp_img), " meters")
        lane_boundaries = self.lane_detector.get_the_polygon_image(warp_img)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.inverted_warper.warp_image(lane_boundaries) 
        # Combine the result with the original image 
        topLeftCornerOfText = (10, 50);
        text = "Curvature radius " + str(np.min(self.lane_detector.get_curvature(warp_img))) + ";"
        cv2.putText(image, text, topLeftCornerOfText, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        offset = self.lane_detector.get_offset(warp_img);
        if offset < 0:
            dir_text = " left"            
        else:
            dir_text = " right"
        text = "Center " +  str(np.abs(offset) ) + " meters" + dir_text

        nextLineOfText = (10, 80);
        cv2.putText(image, text, nextLineOfText, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        return result;