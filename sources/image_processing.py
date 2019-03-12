import lane_detection as ld
import matplotlib.pyplot as plt
import numpy as np
import cv2
import lines
import calibration as cb;
import binary_outputs as bo
import lane_detection as ld

def get_binary(img):
    hls_s_binary = bo.get_hls_s_binary(img, (210, 255));
    scale_x_binary = bo.get_sobel_binary(img, (70, 100));
    yellow_binary = bo.get_yellow_binary(img, (220, 255));
    mag_binary = bo.get_mag_binary(img, 5, (70, 220))
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(hls_s_binary)
    combined_binary[(hls_s_binary == 1) | (scale_x_binary == 1) | (yellow_binary == 1) | (mag_binary == 1)] = 1
    return combined_binary;

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
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        return result;