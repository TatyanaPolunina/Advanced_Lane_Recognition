import numpy as np
import cv2
import lines


class LaneDetector:
    
    def __init__(self, margin = 100):
        self.margin = margin
        self.left_lane = lines.Line()
        self.right_lane = lines.Line()
       
    def find_next_poly(self, binary_warped):
        self.left_lane.find_next_poly(binary_warped, True, self.right_lane);
        self.right_lane.find_next_poly(binary_warped, False, self.left_lane);
         
    def draw_the_lines(self, binary_warped):
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img = self.left_lane.draw_the_line(out_img, [255, 0, 0])
        out_img = self.right_lane.draw_the_line(out_img, [0, 0, 255]); 
        return out_img;
    
    def get_the_polygon_image(self, warped_image):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_lane.bestx, self.left_lane.besty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_lane.bestx, self.right_lane.besty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        return color_warp
    
    def get_curvature(self, warped_image):
        return self.left_lane.radius_of_curvature, self.right_lane.radius_of_curvature
    
    def get_center_position(self, warped_image):
        left_x = lines.get_x(self.left_lane.current_fit, warped_image.shape[0])
        right_x = lines.get_x(self.right_lane.current_fit, warped_image.shape[0])
        return np.int((left_x + right_x) / 2);
    
    def get_offset(self, warped_image,  ym_per_image = 30, xm_per_image = 5.5):
        xm_per_pixel = xm_per_image / warped_image.shape[1];
        offset_in_pixels = self.get_center_position(warped_image) - warped_image.shape[1] /2;
        return offset_in_pixels * xm_per_pixel 