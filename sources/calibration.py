import numpy
import cv2
import matplotlib.pyplot as plt

class Chess_calibration_points:
    obj_points = [] # 3d points in real world space
    img_points = [] # 2d points in image plane.
    img_shape = []
    mtx = []
    dist = []
    def __init__(self, image_names, chess_size):
        objp = numpy.zeros((chess_size[0]*chess_size[1],3), numpy.float32)
        objp[:,:2] = numpy.mgrid[0:chess_size[0],0:chess_size[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        # Step through the list and search for chessboard corners
        for fname in image_names:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if not self.img_shape:
                self.img_shape = gray.shape;

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (chess_size[0],chess_size[1]),None)
            # If found, add object points, image points
            if ret == True:
                self.obj_points.append(objp)
                self.img_points.append(corners)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (chess_size[0],chess_size[1]), corners, ret)
                plt.imshow(img)
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, self.img_shape[::-1], None, None)
        
    def undist_image(self, image):        
        undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        return undist;
        