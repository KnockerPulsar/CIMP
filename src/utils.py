import numpy as np
import pyautogui
import cv2

from typing import List, Tuple
from pynput.mouse import Button
from threaded_capture import ThreadedVideoStream


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


def point_screen_to_image_coordinates(point: Tuple[int, int], image_canvas: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> Tuple[int, int]:

    [canv_x_start, canv_y_start, canv_width, canv_height] = image_canvas
    [image_height, image_width] = image_shape

    dx = point[0] - canv_x_start
    dy = point[1] - canv_y_start

    return (int(dx/canv_width*image_width), int(dy/canv_height*image_height))


def inset_rect(rect: Tuple[int, int, int, int], offset: int) -> Tuple[int, int, int, int]:
    [rect_x1, rect_y1, rect_width, rect_height] = rect
    return (rect_x1 + offset, rect_y1 + offset, rect_width - 2 * offset, rect_height - 2 * offset)


def get_colored_areas(image: np.ndarray, alpha_threshold=0) -> np.ndarray:
    """
    Returns a mask that specified which areas have alpha > alpha_threshold
    Thus, the image must be RGBA.
    """
    assert(len(image.shape) == 3)
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


def draw(pointer_pos: Tuple[int, int], side_len: int, buffer: np.ndarray) -> np.ndarray:
    """
    Draws a square centered at `pointer_pos` and with width/height = `side_length` 
    """

    color = (0, 255, 0, 1)
    start_x = pointer_pos[0] - int(side_len/2)
    end_x = pointer_pos[0] + int(side_len/2)
    start_y = pointer_pos[1] - int(side_len/2)
    end_y = pointer_pos[1] + int(side_len/2)

    buffer[start_y:end_y, start_x:end_x] = color
    return buffer


def init_drawing_buffer(frame_shape: Tuple[int, int, int]) -> np.ndarray:

    # The camera image is RGB, we need the buffer to be RGBA, so 4 channels instead of 3
    # So we can overlay it on top of the image
    buffer_shape = (frame_shape[0], frame_shape[1], frame_shape[2]+1)

    draw_buffer = np.ndarray(shape=buffer_shape)
    draw_buffer[:, :] = (0, 0, 0, 0)
    return draw_buffer

# Gets called whenever the mouse is clicked


def on_click(x, y, button, pressed):
    global Globals

    if(button == Button.left and pressed):
        Globals.draw_command = True
    else:
        Globals.draw_command = False


def clean_up( webcam):
    print("Exiting")
    webcam.release()
    cv2.destroyWindow(Globals.WINDOW_NAME)


class Globals:
    draw_command = False
    WINDOW_NAME = "Results"
