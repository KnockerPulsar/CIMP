from cv2 import bitwise_not, erode, findContours
from cv2 import distanceTransform, ellipse, erode, findContours, threshold
from numpy import histogram, random
from numpy.core.fromnumeric import shape
from numpy.random.mtrand import randint
from scipy.ndimage.filters import median_filter
from scipy.sparse.construct import rand
from skimage.color import rgb2gray
from ui import *
from utils import (
    clean_up,
    get_hand_bbs,
    init,
)
from utils import overlay_images, draw
from time import time
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, closing, erosion
from skimage.draw import ellipse_perimeter, circle_perimeter, disk
from skimage import feature, exposure
import mediapipe as mp

mp_hands = mp.solutions.hands


def main():
    win_name = "Virtual Board?"

    # Start webcam capture thread, setup window
    webcam, draw_buffer = init(win_name)

    loop = True

    # A copy of the previous frame in case the thread hasn't received any new ones
    prev_frame = None

    # For finger detection debugging
    view = 0

    # Running average
    num_fingers_list = []
    num_fingers_window = 15

    # The color we draw in
    # RGBA

    print(
        """
        Press "e" with your hand inside the blue region to start tracking
        Try to keep the background free of highlights or white objects
        """
    )

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 10)

    ms_xstart = int(draw_buffer.shape[1] * 0.25)
    ms_ystart = int(draw_buffer.shape[0] * 0.25)

    ms_width = int(draw_buffer.shape[1] * 0.3)
    ms_height = int(draw_buffer.shape[0] * 0.4)

    x, y, w, h = (
        ms_xstart,
        ms_ystart,
        ms_width,
        ms_height,
    )
    track_window = (x, y, w, h)
    roi_captured = False

    
    sizes = [180, 255]              # How large the the samples for hue and saturation are
    ranges = [0, 180, 0, 255]       # The range of values for hue and saturation
    channels = [0, 1]               # 0-> Hue, 1->Sat.

    roi = None
    roi_hist = None
    backsub = cv2.createBackgroundSubtractorKNN(history = 500, dist2Threshold=150)

    while loop:
        # To calculate FPS
        start_time = time()  # time()

        # Checck if the thread has a new frame
        frame_available, frame = webcam.get_frame()

        # If there's no new frame, use the previous one
        if not frame_available:
            frame = prev_frame

        # flip frame
        frame = cv2.flip(frame, 1)

        fg_mask = backsub.apply(frame)
        fg = cv2.bitwise_and(frame,frame, mask=fg_mask)

        hsv_fg = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV)

        pointer_pos_image_coordinates = (-1, -1)
        draw_command = False
        num_fingers = 0

        # https://stackoverflow.com/questions/8593091/robust-hand-detection-via-computer-vision?noredirect=1&lq=1
        # https://docs.opencv.org/3.4/da/d7f/tutorial_back_projection.html

        # Tracking mode
        # Mean shift seems to be more stable than camshift
        if roi_captured:

            backproj = cv2.calcBackProject(
                [hsv_fg], channels, roi_hist, ranges, scale=3
            )

            ret, track_window = cv2.meanShift(backproj, track_window, term_crit)

            # # Cam shift
            # # pts = cv2.boxPoints(ret)
            # # pts = np.int0(pts)
            # # frame = cv2.polylines(frame, [pts], True, 255, 2)
            # # if(pts.min() == pts.max() == 0):
            # #     roi_captured = False
            
            # Mean shift
            x,y,w,h = track_window
            frame = cv2.rectangle(fg, (x,y), (x+w,y+h), 255,2)


        else:
            # Capture a ROI to use as a search target later on
            key = cv2.waitKey(1) & 0xFF
            if key == ord("e"):
                roi = hsv_fg[y : y + h, x : x + w]

                # Remove the value channel
                roi[:,:,2] = 0

                # For some reason, doing the opposite of 
                # "3. Threshold pixels with low saturation due to their instability."
                # works better
                mask = cv2.inRange(roi,(0,5,0) , (180,50,255))

                roi_hist = cv2.calcHist(
                    [roi], channels, mask, sizes, ranges, accumulate=False
                )
                cv2.normalize(
                    roi_hist, roi_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                )
                # Go into tracking mode
                roi_captured = True

            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

       
        ################################################################################################

        # Paint the buffer on top of the base webcam image
        # frame = overlay_images([frame, draw_buffer])

        # Draw the image and UI
        display_ui(frame, win_name, start_time, num_fingers, display_ui=True)

        # Copy the frame for later use
        prev_frame = frame

        # Check if we want to quit
        loop = check_quit()

    # Clean up
    clean_up(webcam, win_name)


if __name__ == "__main__":
    main()
