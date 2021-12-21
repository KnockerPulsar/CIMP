import numpy as np
from numpy.lib.arraypad import pad
import pyautogui
import cv2

from typing import List, Tuple
from pynput.mouse import Button
from threaded_capture import ThreadedVideoStream

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def init() -> cv2.VideoCapture:
    """
    Only prepares the webcam and creates a window
    Might have more logic later on
    """
    global Globals

    vc = ThreadedVideoStream()
    cv2.namedWindow(Globals.WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(Globals.WINDOW_NAME, 640, 480)

    # Wait until the thread actually starts receiving frames
    frame_available, frame = vc.get_frame()
    while not frame_available:
        frame_available, frame = vc.get_frame()

    # Create the draw buffer
    draw_buffer = init_drawing_buffer(frame.shape)

    return vc, draw_buffer


def get_mouse_position() -> Tuple[int, int]:
    p = pyautogui.position()
    return (p.x, p.y)


def point_inside_canvas(mouse_position: Tuple[int, int], window_rect) -> bool:
    # Top left corner
    x1 = window_rect[0]
    y1 = window_rect[1]

    # Bottom right corner
    x2 = x1 + window_rect[2]
    y2 = y1 + window_rect[3]

    return (x1 <= mouse_position[0] <= x2) and (y1 <= mouse_position[1] <= y2)


def point_screen_to_image_coordinates(
        point: Tuple[int, int], image_canvas: Tuple[int, int, int, int],
        image_shape: Tuple[int, int]) -> Tuple[int, int]:

    [canv_x_start, canv_y_start, canv_width, canv_height] = image_canvas
    [image_height, image_width] = image_shape

    dx = point[0] - canv_x_start
    dy = point[1] - canv_y_start

    return (int(dx / canv_width * image_width),
            int(dy / canv_height * image_height))


def inset_rect(rect: Tuple[int, int, int, int],
               offset: int) -> Tuple[int, int, int, int]:
    [rect_x1, rect_y1, rect_width, rect_height] = rect
    return (rect_x1 + offset, rect_y1 + offset, rect_width - 2 * offset,
            rect_height - 2 * offset)


def get_colored_areas(image: np.ndarray, alpha_threshold=0) -> np.ndarray:
    """
    Returns a mask that specified which areas have alpha > alpha_threshold
    Thus, the image must be RGBA.
    """
    assert (len(image.shape) == 3)
    alpha_exists = np.where(image[:, :, 3] > alpha_threshold)
    return (alpha_exists[0], alpha_exists[1])


def overlay_images(images: List[np.ndarray]) -> np.ndarray:
    """
    Given a list of images with the same width and height, overlays each on over the other
    in the order of the list.
    The first image is considered the base image, must be RGB only.
    The remaining images must be RGBA so that we can check the alpha so that we either overlay or not.
    """
    base_image = images[0]
    for image in images[1:]:
        drawn_to = get_colored_areas(image)

        # Copy RGB channels from the current buffer to the base image
        # AKA replace pixel values
        base_image[drawn_to] = image[drawn_to[0], drawn_to[1], :3]

    return base_image


def draw(pointer_pos: Tuple[int, int], side_len: int,
         buffer: np.ndarray) -> np.ndarray:
    """
    Draws a square centered at `pointer_pos` and with width/height = `side_length` 
    """

    color = (0, 255, 0, 1)
    start_x = pointer_pos[0] - int(side_len / 2)
    end_x = pointer_pos[0] + int(side_len / 2)
    start_y = pointer_pos[1] - int(side_len / 2)
    end_y = pointer_pos[1] + int(side_len / 2)

    buffer[start_y:end_y, start_x:end_x] = color
    return buffer


def init_drawing_buffer(frame_shape: Tuple[int, int, int]) -> np.ndarray:

    # The camera image is RGB, we need the buffer to be RGBA, so 4 channels instead of 3
    # So we can overlay it on top of the image
    buffer_shape = (frame_shape[0], frame_shape[1], frame_shape[2] + 1)

    draw_buffer = np.ndarray(shape=buffer_shape)
    draw_buffer[:, :] = (0, 0, 0, 0)
    return draw_buffer


# Gets called whenever the mouse is clicked


def on_click(x, y, button, pressed):
    global Globals

    if (button == Button.left and pressed):
        Globals.draw_command = True
    else:
        Globals.draw_command = False


def clean_up(webcam):
    print("Exiting")
    webcam.release()
    cv2.destroyWindow(Globals.WINDOW_NAME)


class Globals:
    draw_command = False
    WINDOW_NAME = "Results"


def hsv_threshold(h_lower, h_higher, sat_lower, sat_higher, se_size, frame,
                  prev_frame):
    # Since OpenCV captures images in BGR for some reason
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # attempts at skin detection
    # converting from gbr to hsv color space
    # img_HSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    if cv2.waitKey(1) & 0xFF == ord('w'):
        h_higher += 10
        print(f"h_higher increased {h_higher}")

    if cv2.waitKey(1) & 0xFF == ord('e'):
        h_higher -= 10
        print(f"h_higher decreased {h_higher}")

    if cv2.waitKey(1) & 0xFF == ord('s'):
        h_lower += 10
        print(f"h_lower increased {h_lower}")

    if cv2.waitKey(1) & 0xFF == ord('d'):
        h_lower -= 10
        print(f"h_lower decreased {h_lower}")

    if cv2.waitKey(1) & 0xFF == ord('x'):
        se_size += 2
        print(f"se_size increased {se_size}")

    if cv2.waitKey(1) & 0xFF == ord('c'):
        if (se_size > 2):
            se_size -= 2
            print(f"se_size decreased {se_size}")

    if cv2.waitKey(1) & 0xFF == ord('r'):
        sat_higher += 10
        print(f"sat_higher decreased {sat_higher}")

    if cv2.waitKey(1) & 0xFF == ord('t'):
        sat_higher -= 10
        print(f"sat_loer decreased {sat_higher}")

    if cv2.waitKey(1) & 0xFF == ord('f'):
        sat_lower += 10
        print(f"sat_lower decreased {sat_lower}")

    if cv2.waitKey(1) & 0xFF == ord('g'):
        sat_lower -= 10
        print(f"sat_loer decreased {sat_lower}")

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # img_ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)

    # skin color range for hsv color space
    HSV_mask = cv2.inRange(img_hsv, (h_lower, sat_lower, 50),
                           (h_higher, sat_higher, 200))
    # HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # converting from gbr to YCbCr color space
    # YCrCb_mask = cv2.inRange(img_ycrcb, (0, 135, 85), (255,180,135))
    # YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # merge skin detection (YCbCr and hsv)
    # global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask = HSV_mask
    # global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.medianBlur(global_mask, ksize=se_size)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_CLOSE,
                                   np.ones((se_size * 2, se_size), np.uint8))
    # global_mask = cv2.GaussianBlur(global_mask,sigmaX=3,ksize=[5,5])

    # HSV_result = cv2.bitwise_not(HSV_mask)
    # YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result = cv2.bitwise_not(global_mask)
    # global_result = global_mask
    # global_result=cv2.bitwise_not(global_result)

    # temp = np.zeros(shape=frame.shape, dtype=np.uint8)
    # temp[global_result] = frame[global_result]

    # temp = np.stack((global_result,glo6bal_result,global_result), axis=2)
    # frame = cv2.bitwise_and(frame,temp)  # cv2.bitwise_and(frame,temp)

    if (prev_frame is None):
        prev_frame = frame

    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

    # frame_gray = cv2.medianBlur(frame_gray, ksize=se_size)
    # # frame_gray = cv2.GaussianBlur(
    # #     frame_gray, ksize=[se_size,se_size], sigmaX=se_size, sigmaY=se_size)
    # prev_frame_gray = cv2.medianBlur(prev_frame_gray, ksize=se_size)
    # # prev_frame_gray = cv2.GaussianBlur(
    # #     prev_frame_gray, ksize=[se_size,se_size], sigmaX=se_size, sigmaY=se_size)

    # g_mask = global_result == 255
    # static_mask = (frame_gray-prev_frame_gray) == 0
    # condition = g_mask * static_mask
    # mask = np.where(condition)

    mask = np.where(global_mask == 0)
    frame[mask] = frame[mask] * 0.2
    return h_lower, h_higher, sat_lower, sat_higher, se_size, frame


def get_hand_bbs(frame, hands, padding=20, draw_on_frame=True):
    """
        Given a frame, detects hand in the image and returns their bounding boxes in image space
        Can add padding onto the bounding boxes if needed.
        Can disable drawing on the frame if needed.
    """

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame.flags.writeable = True

    bboxes = []

    # Draw the hand annotations on the frame.
    if results.multi_hand_landmarks:

        for myHand in results.multi_hand_landmarks:
            xList = []
            yList = []
            for id, lm in enumerate(myHand.landmark):
                xList.append(lm.x)
                yList.append(lm.y)

            h, w, c = frame.shape
            xmin, xmax = max(int(min(xList) * w) - padding,
                             0), min(int(max(xList) * w) + padding, w)

            ymin, ymax = max(int(min(yList) * h) - padding,
                             0), min(int(max(yList) * h) + padding, h)

            # bboxInfo = {"id": id, "bbox": bbox,"center": (cx, cy)}
            bboxes.append(((xmin, ymin), (xmax, ymax)))

            if draw_on_frame:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0),
                              2)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, bboxes
