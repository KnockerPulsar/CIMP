from time import time
from typing import Tuple
import cv2
import numpy as np
from threaded_capture import ThreadedVideoStream
from utils import Globals, point_inside_canvas, inset_rect




def display_ui(image: np.ndarray, pointer_pos: Tuple[int, int], start_time: float, border_offset=10):
    """
    To display the camera image, drawing buffer, and UI from back to front
    Only displays how to quit for now
    Assumes that the given image is in RGB
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display quit text at the top left corner
    cv2.putText(image,
                "Press q to quit",
                org=(20, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(10, 255, 0),
                thickness=1)
    cv2.putText(image,
                f"FPS ~{1/(time()-start_time)}",
                org=(20, 80),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(10, 255, 0),
                thickness=1)

    canvas_rect = cv2.getWindowImageRect(Globals.WINDOW_NAME)
    pointer_inside = point_inside_canvas(pointer_pos, canvas_rect)

    image_frame = inset_rect(
        (0, 0, image.shape[1], image.shape[0]), border_offset)

    if(pointer_inside):
        image = cv2.rectangle(image, image_frame, (0, 255, 0), 5)
    else:
        image = cv2.rectangle(image, image_frame, (0, 0, 255), 5)

    # Display the image in the window
    cv2.imshow(Globals.WINDOW_NAME, image)


def check_selection() -> bool:
    """
    Checks if the user pressed q to quit
    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

    return True
