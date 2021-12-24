from cv2 import dilate, erode, findContours
from numpy import unique
from numpy.core.numeric import convolve, count_nonzero, multiply
from ui import *
from utils import clean_up, get_hand_bbs, get_mouse_position, init, init_drawing_buffer, point_screen_to_image_coordinates, hsv_threshold
from pynput.mouse import Listener
from utils import overlay_images, draw, Globals, on_click
from time import perf_counter, time

from skimage.morphology import skeletonize
from skimage.draw import rectangle_perimeter, ellipse, disk, circle_perimeter, ellipse_perimeter
import mediapipe as mp
from scipy.signal import convolve2d

mp_hands = mp.solutions.hands


def finger_detection_options(bitwise_and_or, do_skeletonize, do_threshold):
    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        bitwise_and_or = not bitwise_and_or
        print(f'bitwise_and_or toggled {bitwise_and_or}')
    elif key == ord('s'):
        do_skeletonize = not do_skeletonize
        print(f'do_skeletonize toggled {do_skeletonize}')
    elif key == ord('x'):
        do_threshold = not do_threshold
        print(f'do_threshold toggled {do_threshold}')

    return bitwise_and_or, do_skeletonize, do_threshold


def thresholdHandYCbCr(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image[:, :, 0] = 0
    image = cv2.medianBlur(image, 15)
    new_frame1 = np.ones(image.shape[:2], dtype=np.uint8)
    #new_frame2 = np.ones(image.shape[:2], dtype=np.uint8)
    window_size = 20
    center_x = int(image.shape[0] / 2)
    center_y = int(image.shape[1] / 2)
    skin_cr = image[center_x - window_size:center_x + window_size, center_y - window_size:center_y + window_size, 1]
    # skin_cb = image[center_x - window_size:center_x + window_size, center_y - window_size:center_y + window_size, 2]
    # skin_cb = np.mean(skin_cb)
    skin_cr = np.mean(skin_cr)
    #cr threshholding
    new_frame1[np.where(image[:, :, 1] > (skin_cr + 7))] = 0
    new_frame1[np.where(image[:, :, 1] < (skin_cr - 7))] = 0
    new_frame1[np.where(new_frame1[:, :] == 1)] = 255
    #cb threshholding
    # new_frame2[np.where(image[:, :, 2] > (skin_cb + 7))] = 0
    # new_frame2[np.where(image[:, :, 2] < (skin_cb - 7))] = 0
    # new_frame2[np.where(new_frame2[:, :] == 1)] = 255
    # new_frame[np.where(new_frame1[:, :] == new_frame2[:, :])] = 255
    # new_frame[np.where(image[:, :, 1] > skin_cr - 20 and image[:, :, 1] < skin_cr + 20)] = 255
    # new_frame[np.where(image[:,:, 1] < 120)] = 255
    new_frame = 255 - new_frame1
    return new_frame


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

    # For finger detection debugging
    bitwise_and_or, do_skeletonize, do_threshold = False, False, True
    num_fingers = 0
    num_fingers_list = [0]
    num_fingers_window = 120

    print(
        "\nPress x to toggle YCbCr thresholding (Disables skeletonization and ANDing/ORing)"
        "\nPress s to toggle skeletonization (Disables ANDing/ORing)"
        "\nPress w to toggle circle and skeleton ANDing or ORing"
        "\nA blue background works best"
        )

    with Listener(on_click=on_click) as listener, mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            max_num_hands=1) as hands:  # Listens for mouse events
        while loop:
            # To calculate FPS
            start_time = time()  # time()

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

            # Get hand(s) bounding box
            # Uses mediapipe's hand detector
            frame, hand_bbs = get_hand_bbs(frame, hands)

            for (xmin, ymin), (xmax, ymax) in hand_bbs:

                bitwise_and_or, do_skeletonize, do_threshold = finger_detection_options(
                    bitwise_and_or,
                    do_skeletonize,
                    do_threshold,
                )

                hand_erosion_se_size = 7
                skeleteon_dilation_se_size = 5
                skeleton_intersection_conv_size = 25

                # Threshold image based on YCbCr
                # ONLY WORKS WITH BLUE BACKGROUNDS
                # OFF-WHITE OR ORANGE PERFORM BADLY
                if (do_threshold):
                    roi = thresholdHandYCbCr(frame[ymin:ymax, xmin:xmax])

                    # Calculate elipse width, height, center_x, and center_y
                    [c, r, x, y] = [
                        int((xmax - xmin) * 0.35),
                        int((ymax - ymin) * 0.35), (xmin + xmax) // 2 + 20,
                        (ymin + ymax) // 2 + 20
                    ]

                    # Erode away some of the image to prevent unwanted blobs
                    roi = erode(
                        roi,
                        np.ones((
                            hand_erosion_se_size,
                            hand_erosion_se_size,
                        ), np.uint8))
                    if (do_skeletonize):
                        roi[roi == 255] = True
                        roi = (255 * skeletonize(roi)).astype(np.uint8)

                        # Just an attempt at improving finger intersection tests
                        # Convolve so we can find the middle of the wrist
                        # centre_roi = convolve2d(
                        #     roi,
                        #     np.full((skeleton_intersection_conv_size,
                        #             skeleton_intersection_conv_size), 1))
                        # # The spot with the maximum intersection (and thus maximum value)
                        # should be the wrist (where all lines sprout from)
                        # centre_row = np.amax(centre_roi, 0)
                        # centre_col = np.amax(centre_roi, 1)
                        # Draw an ellipse at that center
                        # circle = ellipse_perimeter(
                        #     y,
                        #     x,
                        #     centre_row[0],
                        #     centre_col[0],
                        #     shape=frame.shape[:2],
                        # )
                        circle = ellipse_perimeter(y,
                                                   x,
                                                   r,
                                                   c,
                                                   shape=frame.shape[:2])

                        # Create an empty buffer
                        imageCircle = np.zeros(frame.shape[:2], dtype=np.uint8)
                        # Using the defined masks (circle or ellipse), fill the buffer at those places
                        imageCircle[circle] = 255

                        # In case the ellipse/circle is defined relative to the whole image
                        # Crop just the place where the ellipse/circle is drawn
                        roi_circle = imageCircle[ymin:ymax, xmin:xmax]

                        # And the ellipse/circle to find the intersection with fingers.
                        if (bitwise_and_or):
                            roi = cv2.bitwise_and(roi, roi_circle)
                        else:
                            roi = cv2.bitwise_or(roi, roi_circle)

                        # An attempt at getting better intersection estimates
                        # Trying to merge smaller intersections from the same finger into one blob
                        roi = dilate(
                            roi,
                            np.ones(shape=(
                                skeleteon_dilation_se_size,
                                skeleteon_dilation_se_size,
                            )))

                        # Get the number of contours and thus the number of points detected
                        # Best case scenario: the number of contours is equal to the number of raised fingers
                        cont, _ = findContours(roi,
                                               mode=cv2.RETR_EXTERNAL,
                                               method=cv2.CHAIN_APPROX_NONE)

                        # Trying to average the number of fingers detected
                        # -1 since the wrist is counted as a contour
                        num_contours = len(cont) - 1
                        num_contours = num_contours if (
                            num_contours >= 1) else 0

                        num_fingers_list.append(num_contours)

                        if (len(num_fingers_list) > num_fingers_window):
                            num_fingers_list.pop(0)

                        num_fingers = round(np.mean(num_fingers_list))

                        frame[ymin:ymax, xmin:xmax] = np.stack((roi, roi, roi),
                                                               axis=2)
                    else:
                        frame[ymin:ymax, xmin:xmax] = np.stack((roi, roi, roi),
                                                               axis=2)
            ################################################################################################
            pointer_inside = point_inside_canvas(pointer_pos, canvas)
            if Globals.draw_command and pointer_inside:
                draw_buffer = draw(pointer_pos_image_coordinates, 10,
                                   draw_buffer)

            # # Paint the buffer on top of the base webcam image
            frame = overlay_images([frame, draw_buffer])

            # Draw the image and UI
            display_ui(frame, pointer_pos, start_time, num_fingers)

            # Copy the frame for later use
            prev_frame = frame

            # Check if we want to quit
            loop = check_selection()

    # Clean up
    clean_up(webcam)


if __name__ == "__main__":
    main()
