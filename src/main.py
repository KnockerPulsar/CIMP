from cv2 import erode, findContours
from cv2 import distanceTransform, ellipse, erode, findContours, threshold
from numpy import random
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
from skimage import feature,exposure
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
        "\nPress the number keys to view different stages of finger detection"
        "\nThreshilding, skeletonization, anding, ording, etc..."
    )
    ######## OpenCv background subtractor##################
    #backSub = cv2.createBackgroundSubtractorKNN(history=30)

    #framesList = []
    while loop:
        # To calculate FPS
        start_time = time()  # time()

        # Change what's shown inside the hand's bounding box

        # Checck if the thread has a new frame

        frame_available, frame = webcam.get_frame()
        ###########################################
        ######## manual background sub ############
        ###########################################

        # framesList.append(frame)
        # if len(framesList):
        # framesList.pop(0)
        # framesList=np.array(framesList)
        # mean_frame = (np.mean(framesList, axis=tuple(range(framesList.ndim - 1))) * 255).astype(np.uint8)
        # frame = mean_frame
        # frame = backSub.apply(frame)
        ###########################################
        ######### HOG descriptror #################
        ###########################################
        fd,frame = feature.hog(
            frame,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=True,
            channel_axis=-1,
        )

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    ms_xstart = int(draw_buffer.shape[1] * 0.25)
    ms_ystart = int(draw_buffer.shape[0] * 0.25)

    ms_width = int(draw_buffer.shape[1] * 0.2)
    ms_height = int(draw_buffer.shape[0] * 0.2)

    x, y, w, h = (
        ms_xstart,
        ms_ystart,
        ms_width,
        ms_height,
    )
    track_window = (x, y, w, h)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        max_num_hands=1,
    ) as hands:  # Listens for mouse events
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

            pointer_pos_image_coordinates = (-1, -1)
            draw_command = False
            num_fingers = 0

            # Get hand(s) bounding box
            # Uses mediapipe's hand detector

            # https://stackoverflow.com/questions/8593091/robust-hand-detection-via-computer-vision?noredirect=1&lq=1
            # https://docs.opencv.org/3.4/da/d7f/tutorial_back_projection.html

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hs_frame = np.empty(hsv_frame.shape, dtype=np.uint8)
            cv2.mixChannels(hsv_frame, hs_frame, [0, 0])
            cv2.mixChannels(hsv_frame, hs_frame, [1, 1])

            sizes = [180, 255]
            ranges = [0, 180, 0, 255]
            channels = [0, 1]

            sat_thresh = 0
            hsv_frame[np.where(hsv_frame[:, :, 1] < sat_thresh)] = (0, 0, 0)

            hs_hist = cv2.calcHist(
                hsv_frame, channels, None, sizes, ranges, accumulate=False
            )
            hs_hist = cv2.normalize(
                hs_hist, hs_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )

            backproj = cv2.calcBackProject(
                [hsv_frame], channels, hs_hist, ranges, scale=1
            )

            ret, track_window = cv2.CamShift(backproj, track_window, term_crit)

            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)

            frame = cv2.polylines(frame, [pts], True, 255, 2)

            ################################################################################################

        # If there's no new frame, use the previous one
        if not frame_available:
            frame = prev_frame

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
