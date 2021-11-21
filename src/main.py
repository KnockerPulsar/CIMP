import cv2
import numpy as np


WINDOW_NAME = "Results"




def init() -> cv2.VideoCapture:
    """
    Only prepares the webcam and creates a window
    Might have more logic later on
    """
    vc = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    return vc


def display_ui(image: np.ndarray):
    """
    To display the camera image, drawing buffer, and UI from back to front
    Only displays how to quit for now
    Assumes that the given image is in RGB
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display quit text at the top left corner
    cv2.putText(image,
                "Press q to quit",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(10, 255, 0),
                thickness=2)

    # Display the image in the window
    cv2.imshow(WINDOW_NAME, image)


def check_selection() -> bool:
    """
    Checks if the user pressed q to quit
    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

    return True


def main():
    webcam = init()
    loop = True

    while loop:

        _, frame = webcam.read()

        # Since OpenCV captures images in BGR for some reason
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        display_ui(frame)
        loop = check_selection()

    print("Exiting")
    webcam.release()
    cv2.destroyWindow("Results")


if __name__ == "__main__":
    main()
