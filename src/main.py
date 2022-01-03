from cv2 import erode, findContours
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


    while loop:
        # To calculate FPS
        start_time = time()  # time()

        # Change what's shown inside the hand's bounding box

        # Checck if the thread has a new frame
        frame_available, frame = webcam.get_frame()

        # flip frame
        frame = cv2.flip(frame, 1)

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
