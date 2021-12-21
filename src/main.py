from ui import *
from utils import clean_up, get_hand_bbs, get_mouse_position, init, init_drawing_buffer, point_screen_to_image_coordinates, hsv_threshold
from pynput.mouse import Listener
from utils import overlay_images, draw, Globals, on_click
from time import time

from skimage.morphology import skeletonize

import mediapipe as mp

mp_hands = mp.solutions.hands

# (Tarek) TODO: Perhaps add linear interpolation fill gaps when the mouse moves too much?


def my_unit_circle(r):
    d = 2 * r + 1
    rx, ry = d / 2, d / 2
    x, y = np.indices((d, d))
    return (np.abs(np.hypot(rx - x, ry - y) - r) < 0.5).astype(int)


def main():
    global Globals  # Holds all global variables

    # Start webcam capture thread, setup window
    webcam, draw_buffer = init()

    # So we can break the main loop
    # Yes, we can also use `brake`, but having the input checks in a fucntion
    # seems neater.
    loop = True

    # A copy of the previous frame in case the thread hasn't received any new ones
    prev_frame = None
    # h_lower = 505
    # h_higher = 177
    # se_size = 5

    h_lower = 80
    h_higher = 160

    sat_lower = 0 * 255
    sat_higher = 1 * 255

    se_size = 5

    with Listener(on_click=on_click) as listener, mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            max_num_hands=2) as hands:  # Listens for mouse events
        while loop:
            # To calculate FPS
            start_time = time()

            # Checck if the thread has a new frame
            frame_available, frame = webcam.get_frame()

            # flip frame
            frame = cv2.flip(frame, 1)

            # If there's no new frame, use the previous one
            if not frame_available:
                frame = prev_frame

            # Get image rect (top left x&y + width and height) in screen space coordinates
            canvas = cv2.getWindowImageRect(Globals.WINDOW_NAME)

            # Get pointer (mouse for now) position in screenspace coordinates
            # If your screen is 1920x1080 pixels, your mouse coodinates are in that range.
            pointer_pos = get_mouse_position()

            # Convert the screenspace coordinates into canvas/image space coordinates
            # AKA get where the pointer is relative to the top left of the canvas.
            pointer_pos_image_coordinates = point_screen_to_image_coordinates(
                pointer_pos, canvas, (frame.shape[0], frame.shape[1]))

            # h_lower, h_higher, sat_lower, sat_higher, se_size, frame = hsv_threshold(h_lower, h_higher, sat_lower, sat_higher,
            #                       se_size, frame, prev_frame)

            frame, hand_bbs = get_hand_bbs(frame, hands)
            for (xmin, ymin), (xmax, ymax) in hand_bbs:
                roi = cv2.cvtColor(frame[ymin:ymax, xmin:xmax],
                                   cv2.COLOR_BGR2GRAY)
                (T,
                 roi) = cv2.threshold(roi, 0, 255,
                                      cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                roi = cv2.medianBlur(roi, 9)
                frame[ymin:ymax, xmin:xmax] = np.stack((roi, roi, roi), axis=2)

            ################################################################################################
            pointer_inside = point_inside_canvas(pointer_pos, canvas)
            if Globals.draw_command and pointer_inside:
                draw_buffer = draw(pointer_pos_image_coordinates, 10,
                                   draw_buffer)

            # # Paint the buffer on top of the base webcam image
            frame = overlay_images([frame, draw_buffer])

            # Draw the image and UI
            display_ui(frame, pointer_pos, start_time)

            # Copy the frame for later use
            prev_frame = frame

            # Check if we want to quit
            loop = check_selection()

    # Clean up
    clean_up(webcam)


if __name__ == "__main__":
    main()
