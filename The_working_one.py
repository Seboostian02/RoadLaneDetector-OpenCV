import cv2
import numpy as np
import time

# 1 - b
cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')

left_top = 0
left_bottom = 0
right_top = 0
right_bottom = 0
while True:
    # 1 - c
    ret, frame = cam.read()
    if ret is False:
        break
    # 2 - a, b
    # scale to 40 percent of the original size (one 12nd of the screen)
    scale_percent = 25
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim)
    orig_copy = resized.copy()

    # 3
    # add grayscale to the video
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale', gray)
    # 4
    # create a black image
    imge = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)
    upper_left = (width * 0.55, height * 0.77)
    upper_right = (width * 0.45, height * 0.77)
    lower_left = (0, height)
    lower_right = (width, height)
    pts = np.array([upper_left, upper_right, lower_left, lower_right], dtype=np.int32)
    cv2.fillConvexPoly(imge, pts, 1)

    # 5 - a
    trapezoid_bounds = (upper_left, upper_right, lower_left, lower_right)
    trapezoid_bounds = np.float32(trapezoid_bounds)
    frame_bounds = ((width, 0), (0, 0), (0, height), (width, height))
    frame_bounds = np.float32(frame_bounds)
    # 5 - b
    magic_matrix = cv2.getPerspectiveTransform(trapezoid_bounds, frame_bounds)
    # 5 - c
    imgegr = imge * gray
    cv2.imshow('masked', imgegr)
    warp = cv2.warpPerspective(imgegr, magic_matrix, (width, height))
    cv2.imshow('TopDown View', warp)
    # 6 - a
    blur = cv2.blur(warp, ksize=(7, 7))
    cv2.imshow('Blurred TopDown', blur)
    # 7 - a
    sobel_vertical = np.float32([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
    sobel_horizontal = np.transpose(sobel_vertical)
    # 7 - b
    frame32_1 = np.float32(blur)
    frame32_2 = frame32_1
    frame32_1 = cv2.filter2D(frame32_1, -1, sobel_vertical)
    frame32_2 = cv2.filter2D(frame32_2, -1, sobel_horizontal)
    # 7 - c
    frame32 = np.sqrt(frame32_1 * frame32_1 + frame32_2 * frame32_2)
    frame8 = cv2.convertScaleAbs(frame32)
    cv2.imshow('Sobel', frame8)
    # 8
    thres = int(255 / 2 - 25)
    rr, img_trs = cv2.threshold(frame8, thres, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary frame', img_trs)
    # 9 - a
    copy_frame = img_trs.copy()
    columnsToDelete = int(width * 0.05)
    copy_frame[:, :columnsToDelete] = 0
    copy_frame[:, width - columnsToDelete:] = 0
    # 9 - b
    # cut the image in half
    width_cutoff = width // 2
    leftFrame = copy_frame[:, :width_cutoff]
    rightFrame = copy_frame[:, width_cutoff:]
    leftPoints = np.argwhere(leftFrame > 0)
    rightPoints = np.argwhere(rightFrame > 0)
    left_xs = leftPoints[:, 1]
    left_ys = leftPoints[:, 0]
    right_xs = rightPoints[:, 1] + width_cutoff
    right_ys = rightPoints[:, 0]

    # 10 - a
    leftLinePoints = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    rightLinePoints = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)
    # 10 - b
    # Calculate intersection points for left line
    y_top_left = 0
    y_bottom_left = height
    x_top_left = (y_top_left - leftLinePoints[0]) / leftLinePoints[1]
    x_bottom_left = (y_bottom_left - leftLinePoints[0]) / leftLinePoints[1]
    # Calculate intersection points for right line
    y_top_right = 0
    y_bottom_right = height
    x_top_right = (y_top_right - rightLinePoints[0]) / rightLinePoints[1]
    x_bottom_right = (y_bottom_right - rightLinePoints[0]) / rightLinePoints[1]

    if (-10 ** 8 <= x_top_left <= 10 ** 8) and (-10 ** 8 <= x_bottom_left <= 10 ** 8):
        left_top = (int(x_top_left), y_top_left)
        left_bottom = (int(x_bottom_left), y_bottom_left)

    if (-10 ** 8 <= x_top_right <= 10 ** 8) and (-10 ** 8 <= x_bottom_right <= 10 ** 8):
        right_top = (int(x_top_right), y_top_right)
        right_bottom = (int(x_bottom_right), y_bottom_right)

    cv2.line(img_trs, left_top, left_bottom, color=(200, 0, 0), thickness=2)
    cv2.line(img_trs, right_top, right_bottom, color=(100, 0, 0), thickness=2)

    cv2.imshow('Binary framw w lines', img_trs)
    # 11 - a
    blank_frame_left = np.zeros((height, width), dtype=np.uint8)
    blank_frame_right = np.zeros((height, width), dtype=np.uint8)
    # 11 - b
    cv2.line(blank_frame_left, left_top, left_bottom, color=(255, 0, 0), thickness=3)
    cv2.line(blank_frame_right, right_top, right_bottom, color=(255, 0, 0), thickness=3)
    # 11 - c
    magic_matrix = cv2.getPerspectiveTransform(frame_bounds, trapezoid_bounds)
    # 11 - d
    rewarp_left = cv2.warpPerspective(blank_frame_left, magic_matrix, (width, height))
    rewarp_right = cv2.warpPerspective(blank_frame_right, magic_matrix, (width, height))
    # 11 - e
    leftFinalPoints = np.argwhere(rewarp_left > 0)
    rightFinalPoints = np.argwhere(rewarp_right > 0)
    orig_copy[leftFinalPoints[:, 0], leftFinalPoints[:, 1]] = [50, 50, 250]
    orig_copy[rightFinalPoints[:, 0], rightFinalPoints[:, 1]] = [50, 250, 50]
    cv2.imshow('Original', orig_copy)
    # if(cv2.waitKey(0)& 0xFF==ord('q')):
    #     break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()