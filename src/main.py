from cv2 import dilate, erode, findContours
from numpy import unique
from numpy.core.numeric import convolve, count_nonzero
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

            # h_lower, h_higher, sat_lower, sat_higher, se_size, frame = hsv_threshold(h_lower, h_higher, sat_lower, sat_higher,
            #                       se_size, frame, prev_frame)
            def thresholdHand(image):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                image[:, :, 0] = 0
                image = cv2.medianBlur(image, 21)
                new_frame = np.zeros(image.shape[:2], dtype=np.uint8)
                new_frame[np.where(image[:, :, 1] < 120)] = 255
                new_frame = 255 - new_frame
                return new_frame

            frame, hand_bbs = get_hand_bbs(frame, hands)
            for (xmin, ymin), (xmax, ymax) in hand_bbs:
                roi = thresholdHand(frame[ymin:ymax, xmin:xmax])
                [c, r, x, y] = [
                    int((xmax - xmin) * 0.35),
                    int((ymax - ymin) * 0.35), (xmin + xmax) // 2+20,
                    (ymin + ymax) // 2 +20
                ]
                kernel = np.ones((5, 5), np.uint8)
                roi = erode(roi, kernel)
                roi[roi == 255] = True
                roi = (255 * skeletonize(roi)).astype(np.uint8)
                window=np.full((25,25),1)
                centre_roi=convolve2d(roi,window)
                centre_row=np.amax(centre_roi,0)
                centre_col=np.amax(centre_roi,1)
                circle = ellipse_perimeter(y, x, centre_row[0], centre_col[0], shape=frame.shape[:2])

                        #image_frame = inset_rect(
                        #    (0, 0, roi.shape[1], roi.shape[0]), 25)
                imageCircle = np.zeros(frame.shape[:2], dtype=np.uint8)
                imageCircle[circle] = 255
                        #image = cv2.rectangle(imageCircle, image_frame,  255, 1)
                roi_circle = imageCircle[ymin:ymax, xmin:xmax]
                roi = cv2.bitwise_and(roi, roi_circle)
                roi = dilate(roi, kernel)
                #cont,h=findContours(roi,mode= cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
                #n_count = len(cont)

                #print(n_count)
                frame[ymin:ymax, xmin:xmax] = np.stack((roi, roi, roi), axis=2)
                #frame=skeletonize(frame)
                #frame = cv2.bitwise_and(frame, imageCircle)
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
