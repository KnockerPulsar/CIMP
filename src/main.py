from cv2 import distanceTransform, ellipse, erode, findContours, threshold
from scipy.ndimage.filters import median_filter
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
from skimage.morphology import binary, skeletonize, closing, erosion, binary_erosion
from skimage.draw import ellipse_perimeter, circle_perimeter, disk, line
import mediapipe as mp

mp_hands = mp.solutions.hands



def finger_detection_views(view):
    key = cv2.waitKey(1) & 0xFF
    for i in range(6):
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

    
    #window showing cr component of YCbCr image
    skin_cr = image[
        center_x - window_size : center_x + window_size,
        center_y - window_size : center_y + window_size,
        1,
    ]

    #window showing hsv components of an image
    skin_hsv = image_hsv[
        center_x - window_size : center_x + window_size,
        center_y - window_size : center_y + window_size,
    ]

    # Detect the colors at the four corners, those colors are very likely to be a background
    # corner_window = 5
    # top_left_corner = (0, 0)
    # top__right_corner = (0, image.shape[1] - 1)
    # bot_left_corner = (image.shape[0] - 1, 0)
    # bot_right_corner = (image.shape[0] - 1, image.shape[1] - 1)

    # bg_clrs = []

    # Get the colors at the four corners using the calculated points & corner window size
    # One second
    # top_left_clrs = image_hsv[
    #     top_left_corner[0] : top_left_corner[0] + corner_window,
    #     top_left_corner[1] : top_left_corner[1] + corner_window,
    #     0
    # ]
    # top_right_clrs = image_hsv[
    #     top__right_corner[0] : top__right_corner[0] + corner_window,
    #     top__right_corner[1] - corner_window : top__right_corner[1],
    #     0
    # ]
    # bot_left_crls = image_hsv[
    #     bot_left_corner[0] - corner_window : bot_left_corner[0],
    #     bot_left_corner[1] : bot_left_corner[1] + corner_window,
    #     0
    # ]
    # bot_right_crls = image_hsv[
    #     bot_right_corner[0] - corner_window : bot_right_corner[0],
    #     bot_right_corner[1] - corner_window : bot_right_corner[1],
    #     0
    # ]

    # top_left_clrs = np.mean(top_left_clrs)
    # top_right_clrs = np.mean(top_right_clrs)
    # bot_left_crls = np.mean(bot_left_crls)
    # bot_right_crls = np.mean(bot_right_crls)

    # bg_clrs.append(np.where(abs(image_hsv - top_left_clrs) <= 10))
    # bg_clrs.append(np.where(abs(image_hsv - top_right_clrs) <= 10))
    # bg_clrs.append(np.where(abs(image_hsv - bot_left_crls) <= 10))
    # bg_clrs.append(np.where(abs(image_hsv - bot_right_crls) <= 10))


    #delta = 30 works well with red background
    delta = 40

    #points around the center point
    # center_x1 = center_x - delta
    # center_x2 = center_x + delta
    # center_y1 = center_y - delta
    # center_y2 = center_y + delta


    # #windows around the 4 new points, each 1 directional
    # skin_cr_x1 = image[
    #     center_x1 - window_size : center_x1 + window_size,
    #     center_y - window_size : center_y + window_size,
    #     1,
    # ]

    # skin_cr_x2 = image[
    #     center_x2 - window_size : center_x2 + window_size,
    #     center_y - window_size : center_y + window_size,
    #     1,
    # ]

    # skin_cr_y1 = image[
    #     center_x - window_size : center_x + window_size,
    #     center_y1 - window_size : center_y1 + window_size,
    #     1,
    # ]

    # skin_cr_y2 = image[
    #     center_x - window_size : center_x + window_size,
    #     center_y2 - window_size : center_y2 + window_size,
    #     1,
    # ]

    # array to store skin colors to average and thrshold on
    # skin_colors = []
    
    skin_cr = np.mean(skin_cr)
    skin_hsv = np.mean(skin_hsv)

    # skin_colors.append(skin_cr)

    # average values along x axis in positive direction and -ve direction, whicherver has the least difference take it
    # average_along_negative_x = np.mean(skin_cr_x1)
    # average_along_positive_x = np.mean(skin_cr_x2)
    # average_along_negative_y = np.mean(skin_cr_y1)
    # average_along_positive_y = np.mean(skin_cr_y2)

    # # skin difference = 5 works well with red background
    # skin_difference = 5
    
    # if abs(skin_cr - average_along_negative_x) < skin_difference :
    #     skin_colors.append(average_along_negative_x)
        
    # if abs(skin_cr - average_along_positive_x) < skin_difference :
    #     skin_colors.append(average_along_positive_x)

    # if abs(skin_cr - average_along_negative_y) < skin_difference :
    #     skin_colors.append(average_along_negative_y)

    # if abs(skin_cr - average_along_positive_y) < skin_difference :
    #     skin_colors.append(average_along_positive_y)

    # skin_cr = np.mean(skin_colors)



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
    skin_cr_threshold = closing(skin_cr_threshold, np.ones((se_window, se_window), np.uint8))

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
    frame, xmin, ymin, xmax, ymax, num_fingers_list, num_fingers_window, dis_trans_list_i, dis_trans_list_j
):

    hand_erosion_se_size = 5

    # Threshold image based on YCbCr
    # ONLY WORKS WITH BLUE BACKGROUNDS
    # OFF-WHITE OR ORANGE PERFORM BADLY
    thresholded = thresholdHandYCbCr(frame[ymin:ymax, xmin:xmax])
    roi = thresholded

    # Calculate elipse width, height, center_x, and center_y
    [c, r, x, y] = [
        int((xmax - xmin) * 0.3),
        int((ymax - ymin) * 0.3),
        (xmin + xmax) // 2,
        (ymin + ymax) // 2,
    ]

    # # Erode away some of the image to prevent unwanted blobs
    roi = erode(
        roi,
        np.ones((hand_erosion_se_size, hand_erosion_se_size), np.uint8),
    )

    roi[roi == 255] = 1

    skeleton = (255 * skeletonize(roi)).astype(np.uint8)

    roi = skeleton


    # TODO: Either use or delete this
    # Test using distance transform
    dis_trans_window = 10

    

    dis_trans = distance_transform_edt(thresholded, return_distances=True)
    palm_center_i, palm_center_j = np.unravel_index(dis_trans.argmax(), dis_trans.shape)

    dis_trans_list_i.append(palm_center_i)
    dis_trans_list_j.append(palm_center_j)
    if len(dis_trans_list_i) > dis_trans_window:
        dis_trans_list_i.remove(dis_trans_list_i[0])
    if len(dis_trans_list_j) > dis_trans_window:
        dis_trans_list_j.remove(dis_trans_list_j[0])
    center_i = int(np.mean(dis_trans_list_i))
    center_j = int(np.mean(dis_trans_list_j))
    # difference_window = 40
    if center_i >= dis_trans.shape[0] or center_j >= dis_trans.shape[1]:
        radius = dis_trans[palm_center_i, palm_center_j]
        test = disk((palm_center_i, palm_center_j), radius * 1.8, shape= dis_trans.shape)
    else:
        radius = dis_trans[center_i, center_j]
        test = disk((center_i, center_j), radius * 1.8, shape= dis_trans.shape)
    dis_trans = np.copy(thresholded)

    dis_trans[test] = 0
    
    # dis_trans = cv2.Sobel(src=frame[ymin:ymax, xmin:xmax], ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    # dis_trans = rgb2gray(dis_trans)
    # rr, cc = np.where(dis_trans != 0)
    # dis_trans = np.copy(thresholded)
    # dis_trans[rr, cc] = 0
    # max_dist = 0
    # max_pos = (-1,-1) 
    # for i in range(len(rr)):
    #     dist = (rr[i]-palm_center_i)**2 + (cc[i]-palm_center_j)**2
    #     if dist > max_dist:
    #         max_dist = dist
    #         max_pos = (rr[i],cc[i])
    # rr, cc = line(palm_center_i, palm_center_j, max_pos[0], max_pos[1])
    # dis_trans = np.copy(thresholded)
    # dis_trans[rr, cc] = 0
    
    # palm_peak_i, palm_peak_j = np.unravel_index(dis_trans_copy.argmin(), dis_trans.shape)
    # rr, cc = line(palm_center_i, palm_center_j, palm_peak_i, palm_center_j)
    # dis_trans_list_i.append(palm_center_i)
    # dis_trans_list_j.append(palm_center_j)
    # if len(dis_trans_list_i) > dis_trans_window:
    #     dis_trans_list_i.remove(dis_trans_list_i[0])
    # if len(dis_trans_list_j) > dis_trans_window:
    #     dis_trans_list_j.remove(dis_trans_list_j[0])
    # center_i = int(np.mean(dis_trans_list_i))
    # center_j = int(np.mean(dis_trans_list_j))
    # radius = dis_trans[palm_center_i, palm_center_j]
    # dis_trans = np.copy(thresholded)
    
    circle = ellipse_perimeter(y, x, r, c, shape=dis_trans.shape)

    
    
    # Create an empty buffer
    imageCircle = np.zeros(frame.shape[:2], dtype=np.uint8)
    # Using the defined masks (circle or ellipse), fill the buffer at those places
    imageCircle[circle] = 255

    # Since the ellipse/circle is defined relative to the whole image
    # Crop just the place where the ellipse/circle is drawn
    roi_circle = imageCircle[ymin:ymax, xmin:xmax]

    # And the ellipse/circle to find the intersection with fingers.
    anded = cv2.bitwise_and(roi, roi_circle)
    ored = cv2.bitwise_or(roi, roi_circle)

    # Contours describe the intersection points better than pixels
    # Because an intersection can be made up of more than 1 pixels
    contours, _ = findContours(
        dis_trans, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )

    # Get the center of each contour remove the outlier
    #  if there's more than one contour
    contour_centers = []
    if len(contours) > 1:
        for cont in contours:
            contour_centers.append((np.mean(int(cont[:, :, 0][0])), np.mean(int(cont[:, :, 1][0]))))

    

    # Reject short branches that are the result of noise
    min_cnt_len = max(c, r) * 2
    too_short = []
    for i, contour in enumerate(contours):
        if cv2.arcLength(contour, False) < min_cnt_len:
            too_short.append(i)
    for element in too_short:
        np.delete(contours, element)

    # Finger count running average
    num_fingers_list.append(len(contours))
    if len(num_fingers_list) > num_fingers_window:
        num_fingers_list.pop(0)

    intersection_positions = np.argwhere(anded == 255)

    return (
        int(np.mean(num_fingers_list)),
        intersection_positions,
        [thresholded, skeleton, anded, ored, dis_trans],
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
    view = 0

    # Running average
    num_fingers_list = []
    num_fingers_window = 15

    # The color we draw in
    # RGBA

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

            # Change what's shown inside the hand's bounding box
            view = finger_detection_views(view)

            # Checck if the thread has a new frame
            frame_available, frame = webcam.get_frame()

            # flip frame
            frame = cv2.flip(frame, 1)

            # If there's no new frame, use the previous one
            if not frame_available:
                frame = prev_frame

            pointer_pos_image_coordinates = (-1, -1)
            draw_command = False
            num_fingers = 0
            pointer_pos = None

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
                (num_fingers, intersection_positions, image_stages,) = get_num_fingers(
                    frame, xmin, ymin, xmax, ymax, num_fingers_list, num_fingers_window, dis_trans_list_i, dis_trans_list_j
                )

                # Based on the number of raised fingers and some more data
                # Either draw or not
                # If you're drawing, draw with a specific color
                draw_command, draw_color, pointer_pos = do_action(
                    num_fingers,
                    intersection_positions,
                    xmin,
                    ymin,
                    draw_buffer,
                    image_stages,
                )

                # WILL go boom if the image stage (skeleton, thresholded, etc...) is not a 2D array
                # 1: thresholded, 2: skeleton, 3: anded, 4: ored
                for i, image_stage in enumerate(image_stages):
                    if view == i + 1:
                        frame[ymin:ymax, xmin:xmax] = np.stack(
                            (image_stage, image_stage, image_stage),
                            axis=2,
                        )

            ################################################################################################

            #print(draw_command)
            if draw_command:
                draw_buffer = draw(
                    pointer_pos, 10, draw_buffer, draw_color
                )

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
