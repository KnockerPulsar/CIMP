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
    with Listener(on_click=on_click) as listener:  # Listens for mouse events
        while loop:
            count = count + 1
            # To calculate FPS
            start_time = time()

            # Checck if the thread has a new frame
            frame_available, frame = webcam.get_frame()

            # flip frame
            frame = cv2.flip(frame,1)

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
            theta = 2.53
            ec_x = 1.6
            ec_y = 2.41
            a = 25.39 
            b = 14.03 
            cb0 = 77
            cr0 = 133
            
            if cv2.waitKey(1) & 0xFF == ord('w'):
                cb0 = cb0+1
                print("cb0 increased")
            if cv2.waitKey(1) & 0xFF == ord('e'):
                cb0 = cb0-1
                print("cb0 decreased")
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cr0 = cr0+1
                print("cr0 increased")
            if cv2.waitKey(1) & 0xFF == ord('d'):
                cr0 = cr0+1
                print("cr0 decreased")

            # dimensions of current frame
            height, width, channels = frame.shape
            # r,g,b = cv2.split(frame)

            frame_YCrCb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
            Y,cr,cb = cv2.split(frame_YCrCb)

            # newFrame=frame

            # if count%30 == 0:
            newFrame = np.zeros(frame.shape)
            # for h in range(0,height):        
            #     for w in range(0,width):
            #         x = np.cos(theta) * (cb[h][w]-cb0) + np.sin(theta) * (cr[h][w] - cr0)
            #         y = -np.sin(theta) * (cb[h][w]-cb0) + np.cos(theta) * (cr[h][w] - cr0)
            #         # print("x= ",x)
            #         # print('y= ',y)
            #         skin_threshold = (((x-ec_x)*(x-ec_x))/(a*a)) + (((y-ec_y)*(y-ec_y))/(b*b))
            #         # print(skin_threshold.shape)
            #         if skin_threshold <= 1:
            #             newFrame[h][w] = 1
            #         else: newFrame[h][w] = 0
            
            x = np.cos(theta) * (cb-cb0) + np.sin(theta) * (cr - cr0)
            y = -np.sin(theta) * (cb-cb0) + np.cos(theta) * (cr - cr0)
            skin_threshold = (((x-ec_x)*(x-ec_x))/(a*a)) + (((y-ec_y)*(y-ec_y))/(b*b))
            # mask = np.zeros((skin_threshold.shape[0],skin_threshold.shape[1]))
            mask = skin_threshold <= 1
            newFrame[mask] = 1
            newFrame *=255
            newFrame = newFrame.astype('uint8')
            # print(newFrame.shape)
            # print(np.amin(cb))
            # print(np.amin(cr))
            # print("shape of mask= ",mask.shape)
            # print("shape of newFrame= ",newFrame.shape)
                    


            ## attempts at skin detection
            # #converting from gbr to hsv color space
            # img_HSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            # #skin color range for hsv color space 
            # HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
            # HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

            # #converting from gbr to YCbCr color space
            # img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)

            # #skin color range for hsv color space 
            # YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
            # YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

            # #merge skin detection (YCbCr and hsv)
            # global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
            # global_mask=cv2.medianBlur(global_mask,3)
            # global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


            # HSV_result = cv2.bitwise_not(HSV_mask)
            # YCrCb_result = cv2.bitwise_not(YCrCb_mask)
            # global_result=cv2.bitwise_not(global_mask)
            # global_result=cv2.bitwise_not(global_result)

            ## different attempt
            # detector = skinDetector(frame)
            # result = detector.find_skin()

            

            # If a draw command is issued, draw in the frame
            # Can be moved to `on_click` but only for mouse clicks. That might reduce drawing lag
            # Once we do the detection ourselves, we'll have to check in the loop anyway
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
