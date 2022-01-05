from cv2 import distanceTransform, ellipse, erode, findContours, putText, threshold
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
from skimage.morphology import binary, skeletonize, opening, closing, erosion, binary_erosion
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
    # added by jimmy while I was writing. Jimmy, please explain
    corner_window = 3
    top_left_corner = (0, 0)
    top_right_corner = (0, image.shape[1] - 1)
    bot_left_corner = (image.shape[0] - 1, 0)
    bot_right_corner = (image.shape[0] - 1, image.shape[1] - 1)

    bg_clrs = []

    # Get the colors at the four corners using the calculated points & corner window size
    # One second
    top_left_clrs = image_hsv[
        top_left_corner[0] : top_left_corner[0] + corner_window,
        top_left_corner[1] : top_left_corner[1] + corner_window,
        0,
    ]
    top_right_clrs = image_hsv[
        top_right_corner[0] : top_right_corner[0] + corner_window,
        top_right_corner[1] - corner_window : top_right_corner[1],
        0,
    ]
    bot_left_crls = image_hsv[
        bot_left_corner[0] - corner_window : bot_left_corner[0],
        bot_left_corner[1] : bot_left_corner[1] + corner_window,
        0,
    ]
    bot_right_crls = image_hsv[
        bot_right_corner[0] - corner_window : bot_right_corner[0],
        bot_right_corner[1] - corner_window : bot_right_corner[1],
        0,
    ]

    top_left_clrs = np.mean(top_left_clrs)
    top_right_clrs = np.mean(top_right_clrs)
    bot_left_crls = np.mean(bot_left_crls)
    bot_right_crls = np.mean(bot_right_crls)

    test_delta = 5

    bg_clrs.append(np.where(abs(image_hsv - top_left_clrs) <= test_delta))
    bg_clrs.append(np.where(abs(image_hsv - top_right_clrs) <= test_delta))
    bg_clrs.append(np.where(abs(image_hsv - bot_left_crls) <= test_delta))
    bg_clrs.append(np.where(abs(image_hsv - bot_right_crls) <= test_delta))

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
    for i in range(len(bg_clrs)):
        skin_cr_threshold[bg_clrs[i][0], bg_clrs[i][1]] = 0
    skin_cr_threshold = closing(
        skin_cr_threshold, np.ones((se_window, se_window), np.uint8)
    )

    # return cv2.bitwise_or(skin_cr_threshold,skin_hsv_threshold)
    return skin_cr_threshold


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


def get_num_fingers(
    frame,
    xmin,
    ymin,
    xmax,
    ymax,
    num_fingers_list,
    num_fingers_window,
    dis_trans_list_i,
    dis_trans_list_j,
):

    img_stages = [frame[ymin:ymax, xmin:xmax]]

    hand_erosion_se_size = 5

    # Threshold image based on YCbCr
    # ONLY WORKS WITH BLUE BACKGROUNDS
    # OFF-WHITE OR ORANGE PERFORM BADLY
    thresholded = thresholdHandYCbCr(frame[ymin:ymax, xmin:xmax])
    img_stages.append(thresholded)

    # Erode away some of the image to prevent unwanted blobs
    roi = erode(
        thresholded,
        np.ones((hand_erosion_se_size, hand_erosion_se_size), np.uint8),
    )

    # Test using distance transform
    dis_trans_window = 5

    se_size = thresholded.size // 5000
    test1 = opening(thresholded, np.ones((se_size, se_size)))

    dis_trans = distance_transform_edt(test1, return_distances=True)
    img_stages.append(dis_trans)


    palm_center_i, palm_center_j = np.unravel_index(dis_trans.argmax(), dis_trans.shape)
    dis_trans_list_i.append(palm_center_i)
    dis_trans_list_j.append(palm_center_j)
    if len(dis_trans_list_i) > dis_trans_window:
        dis_trans_list_i.remove(dis_trans_list_i[0])
    if len(dis_trans_list_j) > dis_trans_window:
        dis_trans_list_j.remove(dis_trans_list_j[0])

    center_i = int(np.mean(dis_trans_list_i))
    center_j = int(np.mean(dis_trans_list_j))

    difference_window = 40
    if center_i >= dis_trans.shape[0] or center_j >= dis_trans.shape[1]:
        radius = dis_trans[palm_center_i, palm_center_j]
        test = disk((palm_center_i, palm_center_j), radius * 1.8, shape=dis_trans.shape)
    else:
        radius = dis_trans[center_i, center_j]
        test = disk((center_i, center_j), radius * 1.8, shape=dis_trans.shape)

    dis_trans = np.copy(thresholded)

    dis_trans[test] = 0

    # Contours describe the intersection points better than pixels
    # Because an intersection can be made up of more than 1 pixels
    contours, _ = findContours(
        dis_trans, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )

    # Get the center of each contour, remove the outlier
    # if there's more than one contour
    # We will assume that the orientation is always vertical
    # given that, we will filter contours below a certain threshold to make decrease number of outliers
    contour_centers = []
    if len(contours) > 1:
        for index, cont in enumerate(contours):
            if np.mean(contours[index]) < palm_center_j:
                contour_centers.append(
                    (np.mean(int(cont[:, :, 0][0])), np.mean(int(cont[:, :, 1][0])))
                )

    for contour_center in contour_centers:
        l = line(center_i, center_j, int(contour_center[1]), int(contour_center[0]))
        print(contour_center[0], contour_center[1])
        try:
            dis_trans[l] = 255
        except:
            print('it happened again')

    img_stages.append(dis_trans)

    wrist_in_img = False
    if len(contour_centers) > 1:
        min_sqr_dist_for_wrist = np.power((ymax - ymin) * 0.7, 2) + np.power(
            (xmax - xmin) * 0.7, 2
        )

        k_means = KMeans(init="k-means++", n_clusters=2, n_init=10)
        k_means.fit(contour_centers)

        cluster1 = k_means.cluster_centers_[0]
        cluster2 = k_means.cluster_centers_[1]

        two_cluster_sqr_dist = np.power(cluster1[0] - cluster2[0], 2) + np.power(
            cluster1[1] - cluster2[1], 2
        )
        if two_cluster_sqr_dist > min_sqr_dist_for_wrist:
            # index = np.where(k_means.labels_ == 0)[0][0]
            # wrist = contour_centers[index]
            # x_img,y_img = int(wrist[0])+xmin, int(wrist[1])+ymin
            print(f"wrist + {random.randint(low=0, high=69)}")
            wrist_in_img = True

    # Gamal tried using this to number the contours, but the numbers kept changing, did not stick to a single contour
    for i in range(len(contour_centers)):
        img_stages[0] = cv2.putText(
            img_stages[0],
            str(i),
            (int(contour_centers[i][0]), int(contour_centers[i][1])),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (135, 50, 168),
            2,
        )

    # Reject short branches that are the result of noise
    # min_cnt_len = max(c, r) * 2
    # too_short = []
    # for i, contour in enumerate(contours):
    #     if cv2.arcLength(contour, False) < min_cnt_len:
    #         too_short.append(i)
    # for element in too_short:
    #     np.delete(contours, element)

    # The number of blobs = number of fingers + wrist blob
    num_blobs = len(contours)

    if wrist_in_img:
        num_blobs -= 1

    # Finger count running average
    num_fingers_list.append(num_blobs)
    if len(num_fingers_list) > num_fingers_window:
        num_fingers_list.pop(0)

    return (
        int(np.mean(num_fingers_list)),
        img_stages,
        contour_centers,
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
    num_fingers_window = 5

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
                (num_fingers, image_stages, contour_centers) = get_num_fingers(
                    frame,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    num_fingers_list,
                    num_fingers_window,
                    dis_trans_list_i,
                    dis_trans_list_j,
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
            # if draw_command and contour_centers:
            #     if num_fingers == 1 and len(contour_centers) == 1:
            #         xpos = contour_centers[0][0] + xmin
            #         ypos = contour_centers[0][1] + ymin
            #         draw_buffer = draw(
            #             (xpos, ypos), 10, draw_buffer, (200, 200, 225, 1.0)
            #         )
            #     elif num_fingers == 2 and len(contour_centers) == 2:
            #         xpos = contour_centers[1][0] + xmin
            #         ypos = contour_centers[1][1] + ymin
            #         draw_buffer = draw(
            #             (xpos, ypos), 10, draw_buffer, (200, 200, 225, 1.0)
            #         )
            #     elif num_fingers > 4:
            #         draw_buffer.fill(0)

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
