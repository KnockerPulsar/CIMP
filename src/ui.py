from time import time
from typing import Tuple
import numpy as np
import cv2
from utils import point_inside_canvas, inset_rect


def display_ui(
    image: np.ndarray,
    win_name: str,
    start_time: float,
    num_fingers: int,
):
    """
    To display the camera image, drawing buffer, and UI from back to front
    Only displays how to quit for now
    Assumes that the given image is in RGB
    """
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Display quit text at the top left corner
    cv2.putText(
        image,
        "Press q to quit",
        org=(20, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(10, 255, 0),
        thickness=1,
    )

    diff = "nan"
    if (time() - start_time) != 0:
        diff = 1 / (time() - start_time)

    cv2.putText(
        image,
        f"FPS ~{diff}",
        org=(20, 80),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(10, 255, 0),
        thickness=1,
    )

    cv2.putText(
        image,
        f'Detected "fingers" {num_fingers}',
        org=(20, 110),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(10, 255, 0),
        thickness=1,
    )

    # Display the image in the window
    cv2.imshow(win_name, image)


def check_quit() -> bool:
    """
    Checks if the user pressed q to quit
    """
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return False

    return True
