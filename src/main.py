from cv2 import erode, findContours, medianBlur
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
from skimage.filters import median
from skimage.morphology import erosion, dilation

mp_hands = mp.solutions.hands


def lab_seg(img):
    tempFrame = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = tempFrame[:, :, 0]
    A = tempFrame[:, :, 1]
    B = tempFrame[:, :, 2]
    H = np.arctan(B / A) * 180 / np.pi
    C = np.sqrt((A * A) + (B * B))

    l_lower = 0
    l_upper = 115

    a_lower = 130
    a_upper = 150

    b_lower = 120
    b_upper = 140

    h_lower = 40
    h_upper = 50

    c_lower = 4
    c_upper = 11

    l_bools = np.where((l_lower < L) & (L < l_upper), 255, 0).astype(np.uint8)
    a_bools = np.where((a_lower < A) & (A < a_upper), 255, 0).astype(np.uint8)
    b_bools = np.where((b_lower < B) & (B < b_upper), 255, 0).astype(np.uint8)
    h_bools = np.where((h_lower < H) & (H < h_upper), 255, 0).astype(np.uint8)
    c_bools = np.where((c_lower < C) & (C < c_upper), 255, 0).astype(np.uint8)


    return l_bools, a_bools, b_bools, h_bools, c_bools


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
    # backSub = cv2.createBackgroundSubtractorKNN(history=30)

    # framesList = []
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
        # fd,frame = feature.hog(
        #    frame,
        #    orientations=8,
        #    pixels_per_cell=(16, 16),
        #    cells_per_block=(1, 1),
        #    visualize=True,
        #    channel_axis=-1,
        # )

        # frame = exposure.rescale_intensity(frame, in_range=(0, 10))
        #####################################################################################
        # flip frame
        # frame = io.imread("./0.jpg")
        kernel=np.full((5,5),1)
        l,a,b, h, c = lab_seg(frame)
        frame = ~h
        #frame=median(dilation(erosion(frame,kernel),kernel))
        frame = cv2.flip(frame, 1)
        # frame = closing(frame, np.full((7, 7), 1))

        # If there's no new frame, use the previous one
        if not frame_available:
            frame = prev_frame

        num_fingers = 0

        # Draw the image and UI
        display_ui(frame, win_name, start_time, num_fingers)

        # Copy the frame for later use
        prev_frame = frame

        # Check if we want to quit
        loop = check_quit()

    # Clean up
    clean_up(webcam, win_name)


if __name__ == "__main__":
    main()
