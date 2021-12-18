from ui import *
from utils import clean_up, get_mouse_position, init, init_drawing_buffer, point_screen_to_image_coordinates
from pynput.mouse import Listener
from utils import overlay_images, draw, Globals, on_click
from time import time

from jeanCV import skinDetector

# (Tarek) TODO: Perhaps add linear interpolation fill gaps when the mouse moves too much?


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
    count = 0
    theta = 2.53
    ec_x = 1.6
    ec_y = 2.41
    a = 25.39
    b = 14.03
    cb0 = 1
    cr0 = 1

    with Listener(on_click=on_click) as listener:  # Listens for mouse events
        while loop:
            count = count + 1
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
                pointer_pos,
                canvas,
                (frame.shape[0], frame.shape[1])
            )

            # Since OpenCV captures images in BGR for some reason
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # (Tarek) TODO: Do image processing, hand detection, and gesture detection from here

            # YCbCr conversion and thresholding
            # the following variables were taken from hand gestures paper
            # cb0 77 -127
            # cr120


            if cv2.waitKey(1) & 0xFF == ord('w'):
                cb0 += 10
                print(cb0)

            if cv2.waitKey(1) & 0xFF == ord('e'):
                cb0 -= 10
                print(cb0)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                cr0 += 10
                print(cr0)

            if cv2.waitKey(1) & 0xFF == ord('d'):
                cr0 -= 10
                print(cr0)

            # dimensions of current frame
            height, width, channels = frame.shape

            frame_YCrCb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
            Y, cr, cb = cv2.split(frame_YCrCb)

            newFrame = np.zeros(frame.shape)

            x = np.cos(theta) * (cb-cb0) + np.sin(theta) * (cr - cr0)
            y = -np.sin(theta) * (cb-cb0) + np.cos(theta) * (cr - cr0)
            skin_threshold = (((x-ec_x)*(x-ec_x))/(a*a)) + \
                (((y-ec_y)*(y-ec_y))/(b*b))
            mask = skin_threshold <= 1
            newFrame[mask] = 1
            newFrame *= 255
            newFrame = newFrame.astype('uint8')

            pointer_inside = point_inside_canvas(pointer_pos, canvas)
            if Globals.draw_command and pointer_inside:
                draw_buffer = draw(pointer_pos_image_coordinates,
                                   10, draw_buffer)

            # Paint the buffer on top of the base webcam image
            frame = overlay_images([frame, draw_buffer])

            # Draw the image and UI
            display_ui(newFrame, pointer_pos, start_time)

            # Copy the frame for later use
            prev_frame = frame

            # Check if we want to quit
            loop = check_selection()

    # Clean up
    clean_up(webcam)


if __name__ == "__main__":
    main()
