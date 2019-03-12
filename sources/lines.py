import numpy as np
import cv2

def generate_x_y_values(img_shape, x_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    fitx = x_fit[0]*ploty**2 + x_fit[1]*ploty + x_fit[2]
    return fitx, ploty

def get_poly(x, y):
    current_fit = np.polyfit(y, x, 2)
    return current_fit

def get_x(fit, y):
    return fit[0]*(y**2) + fit[1]*y + fit[2]

def calculate_histogram(img):
    bottom_half = img[img.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis=0)
    return histogram
    
def get_x_base(img, is_left_line):
    histogram = calculate_histogram(img);
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    if is_left_line:
        return np.argmax(histogram[:midpoint])
    else:
        return np.argmax(histogram[midpoint:]) + midpoint

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, margin = 100):
        # was the line detected in the last iteration?
        self.detected = False  
        #average x values of the fitted line over the last n iterations
        self.bestx = None  
        self.besty = None;
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = []  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None 
        self.margin = margin
        self.number_to_skip = 25
        self.number_of_fails = 25
        
    def find_lane_pixels(self, binary_warped, x_base):
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = np.int(binary_warped.shape[0] * 2 / self.margin);
        # Set minimum number of pixels found to recenter window
        minpix = self.margin / 2
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        x_current = x_base
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height

            win_x_low = x_current - self.margin   # Update this
            win_x_high = x_current + self.margin    # Update this
        
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
         
            # Append these indices to the lists
            lane_inds.append(good_inds)
       
            if (len(good_inds) > minpix):
                x_current = np.int(np.mean(nonzerox[good_inds]));

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds] 

        return x, y


    def fit_polynomial(self, binary_warped, x_base):
        # Find our lane pixels first
        x, y  = self.find_lane_pixels(binary_warped, x_base)
        return np.polyfit(y, x, 2)
        
    def find_next_poly(self, binary_warped, is_left_line, opposite_line):
        if len(self.best_fit) > 0 and (self.number_of_fails < self.number_to_skip) :
            # Grab activated pixels
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            lane_inds = ((nonzerox > (self.best_fit[0]*(nonzeroy**2) + self.best_fit[1]*nonzeroy + 
                        self.best_fit[2] - self.margin)) & (nonzerox < (self.best_fit[0]*(nonzeroy**2) + 
                        self.best_fit[1]*nonzeroy + self.best_fit[2] + self.margin)))

            # Again, extract  line pixel positions
            x = nonzerox[lane_inds]
            y = nonzeroy[lane_inds] 
        
            if (not len(x) or not len(y)):
                self.current_fit = self.fit_polynomial(binary_warped, get_x_base(binary_warped, is_left_line))
            else:
                # Fit new polynomials
                self.current_fit = get_poly(x, y)
            
        else:
            self.current_fit = self.fit_polynomial(binary_warped, get_x_base(binary_warped, is_left_line));
        
        #always fill the first iteration
        if not len(self.best_fit):
            self.best_fit = self.current_fit;
            self.bestx, self.besty = generate_x_y_values(binary_warped.shape, self.best_fit)
            self.number_of_fails = 0;
        
        self.diff = np.abs(self.current_fit - self.best_fit);
        self.line_base_pos = self.calculate_the_distance(binary_warped.shape)
        self.best_fit = self.current_fit    
        self.calculate_curvature(binary_warped)
        self.allx, self.ally = generate_x_y_values(binary_warped.shape, self.current_fit)
        self.detected = self.check_correctness(opposite_line, binary_warped.shape)
        if (self.detected):
            self.best_fit = self.current_fit;
            self.bestx, self.besty = generate_x_y_values(binary_warped.shape, self.best_fit)
            self.number_of_fails = 0;            
        else:
            ++self.number_of_fails;
   
   
    def check_intersection(self, opposite_line, image_shape):
        if not len(opposite_line.best_fit)  or not len(self.current_fit):
            return True;        
        bottom_x = get_x(self.current_fit, image_shape[0]);
        top_x = get_x(self.current_fit, 0);
        bottom_op_x = get_x(opposite_line.best_fit, image_shape[0]);
        top_op_x = get_x(opposite_line.best_fit, 0)
        return np.sign(bottom_x - bottom_op_x) == np.sign(top_x - top_op_x)
        
    def check_correctness(self, opposite_line, image_shape):
        if opposite_line.line_base_pos:
            is_distance_correct = np.abs(self.line_base_pos - opposite_line.line_base_pos)  > 3.4
        else:
            is_distance_correct = np.abs(self.line_base_pos) > 1 and  np.abs(self.line_base_pos)  < 2 
        is_curvature_correct = self.radius_of_curvature > 150;
        line_not_intersected = self.check_intersection(opposite_line, image_shape);
        return  is_distance_correct and is_curvature_correct and line_not_intersected;
    
    def calculate_the_distance(self, image_shape, xm_per_image = 6):
        xm_per_pix = xm_per_image/image_shape[1] # meters per pixel in x dimension
        y = image_shape[0]
        x = get_x(self.current_fit, y);
        return (image_shape[1]/2 - x) * xm_per_pix;
        
    def calculate_curvature(self, warped_image, ym_per_image = 30, xm_per_image = 6):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = ym_per_image/warped_image.shape[0] # meters per pixel in y dimension
        xm_per_pix = xm_per_image/warped_image.shape[1] # meters per pixel in x dimension

        fitx, ploty = generate_x_y_values(warped_image.shape, self.current_fit)
        x =  fitx[::-1]  # Reverse to match top-to-bottom in y
    
        fit_cr = np.polyfit(ploty*ym_per_pix, fitx*xm_per_pix, 2)
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        ym = y_eval * ym_per_pix
        self.radius_of_curvature = (1 + (2 * fit_cr[0] * ym + fit_cr[1]) **2)**1.5 / np.abs(2 * fit_cr[0])

    def draw_the_line(self, out_img, color):
        # Grab activated pixels
        nonzero = out_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        window_img = np.zeros_like(out_img)
        lane_inds = ((nonzerox > (self.best_fit[0]*(nonzeroy**2) + self.best_fit[1]*nonzeroy + 
                    self.best_fit[2] - self.margin)) & (nonzerox < (self.best_fit[0]*(nonzeroy**2) + 
                    self.best_fit[1]*nonzeroy + self.best_fit[2] + self.margin)))
        # Color in line pixels
        out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = color

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        line_window1 = np.array([np.transpose(np.vstack([self.bestx - self.margin, self.besty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.bestx + self.margin, 
                                  self.besty])))])
        line_pts = np.hstack((line_window1, line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
        return cv2.addWeighted(out_img, 1, window_img, 0.3, 0)