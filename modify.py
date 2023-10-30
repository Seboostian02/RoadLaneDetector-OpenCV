import cv2
import numpy as np


def limits(width_in, height_in):
    """
    :param width_in:
    :param height_in:
    :return: arr of limits  [upper_right, upper_left, lower_left, lower_right]
    """
    upper_left = (int(width_in * 0.45), int(height_in * 0.77))
    upper_right = (int(width_in * 0.55), int(height_in * 0.77))
    lower_left = (int(width_in * 0), int(height_in * 1))
    lower_right = (int(width_in * 1), int(height_in * 1))

    arr_of_limits = [upper_right, upper_left, lower_left, lower_right]

    return np.array(arr_of_limits, dtype=np.int32)


def stretch(in_trapez_bounds, in_width, in_height):
    """
    Stretch the frame
    :param in_trapez_bounds: margini trapez
    :param in_width: frame width
    :param in_height: frame height
    :return: screen_bounds and stretched frame
    """
    in_trapez_bounds = np.float32(in_trapez_bounds)
    out_screen_bounds = np.array([(in_width, 0), (0, 0), (0, in_height), (in_width, in_height)], dtype=np.float32)

    perspective_matrix = cv2.getPerspectiveTransform(in_trapez_bounds, out_screen_bounds)

    return out_screen_bounds, cv2.warpPerspective(masked_frame, perspective_matrix, (in_width, in_height))


def sobel(in_stretched_frame):
    # Matrici Sobel
    sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_vertical = np.transpose(sobel_horizontal)

    # resized = resized.astype(np.float32)
    stretched_frame_float32 = np.float32(in_stretched_frame)

    # Aplicare sobel
    horizontal_edges = cv2.filter2D(stretched_frame_float32, -1, sobel_horizontal)
    vertical_edges = cv2.filter2D(stretched_frame_float32, -1, sobel_vertical)

    sobel_filter = np.sqrt(np.square(horizontal_edges) + np.square(vertical_edges))

    return cv2.convertScaleAbs(sobel_filter)


def find_street_markings_coordinates(my_frame, percent_to_remove = 5):
    frame_copy = my_frame.copy()
    height, width = frame_copy.shape

    # procentaj de delete
    num_cols_to_black = int(percent_to_remove/100 * width)

    # face coloanele negre stanga-dreatpta
    frame_copy[:, :num_cols_to_black] = 0
    frame_copy[:, -num_cols_to_black:] = 0

    # gaseste coordonatele pixelilor albi
    left_indices = np.argwhere(frame_copy[:, :width // 2] > 0)
    right_indices = np.argwhere(frame_copy[:, width // 2:] > 0)

    # extrage x si y
    left_ys, left_xs = left_indices[:, 0], left_indices[:, 1]
    right_ys, right_xs = right_indices[:, 0], right_indices[:, 1] + (width // 2)

    return left_xs, left_ys, right_xs, right_ys


def find_and_draw_lane_edges(frame):
    left_ys, left_xs, right_ys, right_xs = find_street_markings_coordinates(frame)

    # Separam pixelii stanga - deapta
    left_pixels = np.column_stack((left_xs, left_ys))
    right_pixels = np.column_stack((right_xs, right_ys))

    # Pozitionare linii
    if len(left_pixels) > 0:
        left_line = np.polynomial.polynomial.polyfit(left_pixels[:, 1], left_pixels[:, 0], 1)
        left_top_y = 0
        left_bottom_y = frame.shape[0] # height
        left_top_x = int((0 - left_line[0]) / left_line[1])
        left_bottom_x = int((frame.shape[0] - left_line[0]) / left_line[1])
    if len(right_pixels) > 0:
        right_line = np.polynomial.polynomial.polyfit(right_pixels[:, 1], right_pixels[:, 0], 1)
        right_top_y = 0
        right_bottom_y = frame.shape[0]  # height
        right_top_x = int((0 - right_line[0]) / right_line[1])
        right_bottom_x = int((frame.shape[0] - right_line[0]) / right_line[1])

    # Verificare puncte proaste
    if abs(left_top_x) > 1e8:
        left_top_x = left_top_x if left_top_x != 0 else left_top_x
    if abs(left_bottom_x) > 1e8:
        left_bottom_x = left_bottom_x if left_bottom_x != 0 else left_bottom_x
    if abs(right_top_x) > 1e8:
        right_top_x = right_top_x if right_top_x != 0 else right_top_x
    if abs(right_bottom_x) > 1e8:
        right_bottom_x = right_bottom_x if right_bottom_x != 0 else right_bottom_x

    # Detalii linii
    left_line_color = (200, 0, 0)
    right_line_color = (100, 0, 0)
    line_width = 5

    # Desenare linii
    cv2.line(frame, (left_top_x, left_top_y), (left_bottom_x,  left_bottom_y), left_line_color, line_width)
    cv2.line(frame, (right_top_x, right_top_y), (right_bottom_x,  right_bottom_y), right_line_color, line_width)

    # Linie de mijloc
    middle_x = frame.shape[1] // 2
    cv2.line(frame, (middle_x, 0), (middle_x, frame.shape[0]), (255, 0, 0), 1)

    left_top = left_top_y, left_top_x
    left_bottom = left_bottom_y, left_bottom_x
    right_top = right_top_y, right_top_x
    right_bottom = right_bottom_y, right_bottom_x

    return frame, left_top, left_bottom, right_top, right_bottom

SCALE_PERCENT = 25
THRESHOLD_VALUE = 255/2 # 127
WHITE_COLOR = (255, 255, 255)

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')

while True:
    ret, frame = cam.read()

    if ret is False:
        break

    new_width = int(frame.shape[1] * SCALE_PERCENT / 100)
    new_height = int(frame.shape[0] * SCALE_PERCENT / 100)

    frame = cv2.resize(frame, (new_width, new_height))
    original = frame.copy()
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = frame.shape[:2]
    mask = np.zeros_like(image_gray, dtype=np.uint8)

    # --------------------------------  creare trapez
    trapez_bounds = limits(width, height)
    cv2.fillConvexPoly(mask, trapez_bounds, WHITE_COLOR)
    # -------------------------------- aplicare masca
    masked_frame = cv2.bitwise_and(image_gray, image_gray, mask=mask)
    # --------------------------------- Stretch
    screen_bounds, stretched_frame = stretch(trapez_bounds, new_width, new_height)
    # --------------------------------- BLUR
    blurred_frame = cv2.blur(stretched_frame, ksize=(5, 5))
    # ----------------------------- SOBEL
    sobel_filter_uint8 = sobel(stretched_frame)
    # ----------------------------- binary thing
    # AICI SCHIMBI FRAME-UL PE CARE FACI
    _, binary_frame = cv2.threshold(stretched_frame, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    # imaginea, valoarea, absolute white, tip de threshold

    # ----------------------------- coordonate si delete la noise
    processed_frame, left_top, left_bottom, right_top, right_bottom = find_and_draw_lane_edges(binary_frame)

    # Display the processed frame

    # ------------------ FINAL?

    # Nu mai merge sa afisez ca in main liniile pe 2 frame-uri diferite

    # final1 = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    # cv2.line(final1, left_top, left_bottom, (255, 0, 0), 3)
    #
    # trapez_bounds = np.float32(trapez_bounds)
    # screen_bounds = np.float32(screen_bounds)
    #
    # matrix = cv2.getPerspectiveTransform(screen_bounds, trapez_bounds)
    # final_lines_left = cv2.warpPerspective(final1, matrix, (new_width, new_height))
    # cv2.imshow('final1', final_lines_left)
    #
    # left_xs1, left_ys1, right_xs1, right_ys1 = find_street_markings_coordinates(final_lines_left)
    # left_line_coords = left_ys1, left_xs1, right_ys1, right_xs1
    # #---------------------------------------------dreapta
    # left_xs, left_ys, right_xs, right_ys = find_street_markings_coordinates(binary_frame)
    # final2 = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    # #
    #
    # cv2.line(final2, right_top, right_bottom, (50, 250, 50), 3)
    # #
    # matrix2 = cv2.getPerspectiveTransform(screen_bounds, trapez_bounds)
    # final_lines_right = cv2.warpPerspective(final2, matrix2, (new_width, new_height))
    # cv2.imshow('final2', final_lines_right)
    #
    # left_xs2, left_ys2, right_xs2, right_ys2 = find_street_markings_coordinates(final_lines_left)
    # right_line_coords = left_ys2, left_xs2, right_ys2, right_xs2
    #
    # result_frame = original.copy()
    # for coord in left_line_coords:
    #       result_frame[coord[0], coord[1]] = [0, 0, 255]  # Red
    # for coord in right_line_coords:
    #      result_frame[coord[0], coord[1]] = [0, 255, 0]  # Green
    #
    # cv2.imshow("Lane Detection", result_frame)


    # ----------------- PANA AICI



    # print
    cv2.imshow('Original', original)
    cv2.imshow('Grayscale', image_gray)
    cv2.imshow('Trapez', mask)
    cv2.imshow('Road', masked_frame)
    cv2.imshow('Top Down', stretched_frame)
    cv2.imshow('Blurred Video', blurred_frame)
    cv2.imshow('Sobel', sobel_filter_uint8)
    cv2.imshow('Binary Image', binary_frame)

    # close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()