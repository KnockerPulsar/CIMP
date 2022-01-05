from cv2 import (
    bitwise_not,
    dilate,
    distanceTransform,
    ellipse,
    erode,
    findContours,
    putText,
    threshold,
)
from numpy import random
from numpy.random.mtrand import randint
from scipy.ndimage.filters import median_filter
from scipy.sparse.construct import rand
from skimage.color import rgb2gray
from ui import *
from utils import (
    clean_up,
    get_hand_bbs,
    init,
)
from utils import overlay_images, draw
from time import time
from scipy.ndimage import distance_transform_edt
from skimage.morphology import (
    binary,
    skeletonize,
    opening,
    closing,
    erosion,
    binary_erosion,
)
from skimage.draw import ellipse_perimeter, circle_perimeter, disk, line
import mediapipe as mp

from sklearn.cluster import KMeans

mp_hands = mp.solutions.hands


def finger_detection_views(view, img_stages):
    key = cv2.waitKey(1) & 0xFF
    for i in range(1, len(img_stages) + 1):
        if key == ord(str(i)):
            print(f"Mode set to {i}")
            return i
    return view


def thresholdHandYCbCr(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

    image[:, :, 0] = 0
    image = cv2.medianBlur(image, 15)

    image_hsv[:, :, 1:3] = 0
    image_hsv = cv2.medianBlur(image_hsv, 15)

    window_size = image.size // 5000
    center_x = image.shape[0] // 2
    center_y = image.shape[1] // 2

    # window showing cr component of YCbCr image
    skin_cr = image[
        center_x - window_size : center_x + window_size,
        center_y - window_size : center_y + window_size,
        1,
    ]

    # window showing hsv components of an image
    skin_hsv = image_hsv[
        center_x - window_size : center_x + window_size,
        center_y - window_size : center_y + window_size,
    ]

    # Detect the colors at the four corners, those colors are very likely to be a background
    # corner_window = 3
    # top_left_corner = (0, 0)
    # top_right_corner = (0, image.shape[1] - 1)
    # bot_left_corner = (image.shape[0] - 1, 0)
    # bot_right_corner = (image.shape[0] - 1, image.shape[1] - 1)

    # bg_clrs = []

    # # Get the colors at the four corners using the calculated points & corner window size
    # # One second
    # top_left_clrs = image_hsv[
    #     top_left_corner[0] : top_left_corner[0] + corner_window,
    #     top_left_corner[1] : top_left_corner[1] + corner_window,
    #     0,
    # ]
    # top_right_clrs = image_hsv[
    #     top_right_corner[0] : top_right_corner[0] + corner_window,
    #     top_right_corner[1] - corner_window : top_right_corner[1],
    #     0,
    # ]
    # bot_left_crls = image_hsv[
    #     bot_left_corner[0] - corner_window : bot_left_corner[0],
    #     bot_left_corner[1] : bot_left_corner[1] + corner_window,
    #     0,
    # ]
    # bot_right_crls = image_hsv[
    #     bot_right_corner[0] - corner_window : bot_right_corner[0],
    #     bot_right_corner[1] - corner_window : bot_right_corner[1],
    #     0,
    # ]

    # top_left_clrs = np.mean(top_left_clrs)
    # top_right_clrs = np.mean(top_right_clrs)
    # bot_left_crls = np.mean(bot_left_crls)
    # bot_right_crls = np.mean(bot_right_crls)

    # test_delta = 5

    # bg_clrs.append(np.where(abs(image_hsv - top_left_clrs) <= test_delta))
    # bg_clrs.append(np.where(abs(image_hsv - top_right_clrs) <= test_delta))
    # bg_clrs.append(np.where(abs(image_hsv - bot_left_crls) <= test_delta))
    # bg_clrs.append(np.where(abs(image_hsv - bot_right_crls) <= test_delta))

    skin_cr = np.mean(skin_cr)
    skin_hsv = np.mean(skin_hsv)

    delta_cr = 10
    delta_hsv = 20
    skin_cr_threshold = cv2.inRange(
        image[:, :, 1], skin_cr - delta_cr, skin_cr + delta_cr
    )
    skin_hsv_threshold = cv2.inRange(
        image_hsv, skin_hsv - delta_hsv, skin_hsv + delta_hsv
    )

    se_window = 5
    # for i in range(len(bg_clrs)):
    #     skin_cr_threshold[bg_clrs[i][0], bg_clrs[i][1]] = 0
    skin_cr_threshold = closing(
        skin_cr_threshold, np.ones((se_window, se_window), np.uint8)
    )

    # return cv2.bitwise_or(skin_cr_threshold,skin_hsv_threshold)
    return skin_cr_threshold


#


def do_action(
    num_fingers, intersection_positions, xmin, ymin, draw_buffer, image_stages
):

    # Protip: WILL kaboom if we add more stages
    [thresholded, skeleton, anded, ored, dis_trans] = image_stages
    draw_command, draw_color, pointer_pos = False, (0, 0, 0, 0), None

    # Get the highest white pixel
    # Should correspons to the raised finger
    if num_fingers == 1 and len(intersection_positions) == 1:

        finger_tip = np.argmin(np.where(skeleton == 255)[1])

        draw_command = True
        pointer_pos = (
            xmin + intersection_positions[0][0],
            ymin + finger_tip,
        )
        draw_color = (0, 255, 0, 1)

    # Try to get the middle between the 2 raised fingers
    elif num_fingers == 2 and len(intersection_positions) == 2:

        draw_command = True
        white_positions_x = (
            intersection_positions[0][0] + intersection_positions[1][0]
        ) // 2
        white_positions_y = (
            intersection_positions[0][1] + intersection_positions[1][1]
        ) // 2
        pointer_pos = (
            xmin + white_positions_x,
            ymin + white_positions_y,
        )
        draw_color = (0, 0, 255, 1)

    # Clear buffer
    elif num_fingers == 5:
        draw_buffer[:, :] = (0, 0, 0, 0)

    return draw_command, draw_color, pointer_pos


def get_num_fingers_morph(
    frame,
    xmin,
    ymin,
    xmax,
    ymax,
    num_fingers_list,
    num_fingers_window,
):

    img_stages = [frame[ymin:ymax, xmin:xmax]]

    hand_erosion_se_size = 5

    # Threshold image based on YCbCr
    # ONLY WORKS WITH BLUE BACKGROUNDS
    # OFF-WHITE OR ORANGE PERFORM BADLY
    thresholded = thresholdHandYCbCr(frame[ymin:ymax, xmin:xmax])
    img_stages.append(thresholded)

    erosion_width = (xmax - xmin) // 8
    erosion_height = (ymax - ymin) // 8
    roi = erode(thresholded, np.ones((erosion_width, erosion_height), np.uint8))
    roi = dilate(
        roi,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, [int(erosion_width * 2), int(erosion_height * 2)]
        ),
    )

    # Remove palm
    fingers_only = img_stages[1] - roi
    img_stages.append(fingers_only)

    # eroding again to get rid of thin outlies at the perimeter of the palm
    # this erosion size would be better if variable
    fingers_only = erode(fingers_only, np.ones((6, 6), np.uint8), iterations=1)
    img_stages.append(fingers_only)

    num_fingers_list = [0]

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = False
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(cv2.bitwise_not(fingers_only))
    cv2.drawKeypoints(img_stages[0], keypoints, img_stages[0])

    num_fingers_list.append(len(keypoints))
    if len(num_fingers_list) > num_fingers_window:
        num_fingers_list.pop(0)

    avg_finger_count = int(np.mean(num_fingers_list))
    # print(avg_finger_count + '//' + len(keypoints))
    print(num_fingers_list)

    return (
        avg_finger_count,
        img_stages,
        num_fingers_list,
        [keypoint.pt for keypoint in keypoints],
    )


def main():
    win_name = "Virtual Board?"
    dis_trans_list_i = []
    dis_trans_list_j = []

    # Start webcam capture thread, setup window
    webcam, draw_buffer = init(win_name)

    loop = True

    # A copy of the previous frame in case the thread hasn't received any new ones
    prev_frame = None

    # For finger detection debugging
    view = 4

    # Running average
    num_fingers_list = []
    num_fingers_window = 10

    print(
        "\nPress the number keys to view different stages of finger detection"
        "\nThreshilding, skeletonization, anding, ording, etc..."
    )

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        max_num_hands=1,
    ) as hands:  # Listens for mouse events
        while loop:
            # To calculate FPS
            start_time = time()  # time()

            # Checck if the thread has a new frame
            frame_available, frame = webcam.get_frame()

            # If there's no new frame, use the previous one
            if not frame_available:
                frame = prev_frame

            # flip frame
            frame = cv2.flip(frame, 1)

            pointer_pos_image_coordinates = (-1, -1)
            draw_command = True
            contour_centers = []
            num_fingers = 0

            # Get hand(s) bounding box
            # Uses mediapipe's hand detector
            frame, hand_bbs = get_hand_bbs(frame, hands)

            ################################################################################################

            # Looping doesn't really matter for now
            if len(hand_bbs) == 1:

                (xmin, ymin), (xmax, ymax) = hand_bbs[0]

                # Thresholds, skeletonizes, and intersects the skeleton with an ellipse
                # Gets you the number of raised fingers and intersection positions
                # Also returns some intermediary images to help with debugging
                (
                    num_fingers,
                    image_stages,
                    num_fingers_list,
                    contour_centers,
                ) = get_num_fingers_morph(
                    frame,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    num_fingers_list,
                    num_fingers_window,
                )

                # Change what's shown inside the hand's bounding box
                view = finger_detection_views(view, image_stages)

                # Based on the number of raised fingers and some more data
                # Either draw or not
                # If you're drawing, draw with a specific color
                # draw_command, draw_color, pointer_pos = do_action(
                #     num_fingers,
                #     ,
                #     xmin,
                #     ymin,
                #     draw_buffer,
                #     image_stages,
                # )

                # WILL go boom if the image stage (skeleton, thresholded, etc...) is not a 2D array
                # 1: thresholded, 2: skeleton, 3: anded, 4: ored
                for i, image_stage in enumerate(image_stages):
                    if view == i + 1:
                        if len(image_stage.shape) == 2:
                            frame[ymin:ymax, xmin:xmax] = np.stack(
                                (image_stage, image_stage, image_stage),
                                axis=2,
                            )
                        else:
                            frame[ymin:ymax, xmin:xmax] = image_stage

            ################################################################################################

            # #print(draw_command)
            # check for number of fingers raised,
            # I thought that drawing at contour_centers[0] or [1] would do the job
            # but apparently the contours are succeptible to noise and keep rotating so 0 and 1 are not ideal
            # we at some point added a condition to check for len(contour_centers) as well
            # but that made the window focus on the wrist without the raised index finger for some reason

            # comment this for now to work on stabilizing the contours
            if draw_command and contour_centers:
                if num_fingers == 1 and len(contour_centers) == 1:
                    xpos = contour_centers[0][0] + xmin
                    ypos = contour_centers[0][1] + ymin
                    draw_buffer = draw(
                        (xpos, ypos), 10, draw_buffer, (200, 200, 225, 1.0)
                    )
                elif num_fingers == 2 and len(contour_centers) == 2:
                    xpos = contour_centers[1][0] + xmin
                    ypos = contour_centers[1][1] + ymin
                    draw_buffer = draw(
                        (xpos, ypos), 10, draw_buffer, (200, 200, 225, 1.0)
                    )
                elif num_fingers > 4:
                    draw_buffer.fill(0)

            # Paint the buffer on top of the base webcam image
            frame = overlay_images([frame, draw_buffer])

            # Draw the image and UI
            display_ui(frame, win_name, start_time, num_fingers)

            # Copy the frame for later use
            prev_frame = frame

            # Check if we want to quit
            loop = check_quit()

    # Clean up
    clean_up(webcam, win_name)


if __name__ == "__main__":
    main()
