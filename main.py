# import cv2
# import numpy as np
#
#
# def limits(width_in, height_in):
#     """
#     :param width_in:
#     :param height_in:
#     :return: arr of limits  [upper_right, upper_left, lower_left, lower_right]
#     """
#     upper_left = (int(width_in * 0.40), int(height_in * 0.77))
#     upper_right = (int(width_in * 0.60), int(height_in * 0.77))
#     lower_left = (int(width_in * 0), int(height_in * 1))
#     lower_right = (int(width_in * 1), int(height_in * 1))
#
#     arr_of_limits = [upper_right, upper_left, lower_left, lower_right]
#
#     return np.array(arr_of_limits, dtype=np.int32)
#
#
# def stretch(in_trapez_bounds, in_width, in_height):
#     """
#     Stretch the frame
#     :param in_trapez_bounds: margini trapez
#     :param in_width: frame width
#     :param in_height: frame height
#     :return: screen_bounds and stretched frame
#     """
#     in_trapez_bounds = np.float32(in_trapez_bounds)
#     out_screen_bounds = np.array([(in_width, 0), (0, 0), (0, in_height), (in_width, in_height)], dtype=np.float32)
#
#     perspective_matrix = cv2.getPerspectiveTransform(in_trapez_bounds, out_screen_bounds)
#
#     return out_screen_bounds, cv2.warpPerspective(masked_frame, perspective_matrix, (in_width, in_height))
#
#
# def sobel(in_stretched_frame):
#     sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
#     sobel_vertical = np.transpose(sobel_horizontal)
#
#     # resized = resized.astype(np.float32)
#     stretched_frame_float32 = np.float32(in_stretched_frame)
#
#     horizontal_edges = cv2.filter2D(stretched_frame_float32, -1, sobel_horizontal)
#     vertical_edges = cv2.filter2D(stretched_frame_float32, -1, sobel_vertical)
#
#     sobel_filter = np.sqrt(np.square(horizontal_edges) + np.square(vertical_edges))
#
#     return cv2.convertScaleAbs(sobel_filter)
#
#
# def find_street_markings_coordinates(my_frame, percent_to_remove = 5):
#     frame_copy = my_frame.copy()
#     height = int(frame_copy.shape[0])
#     width = int(frame_copy.shape[1])
#
#     # procentaj de delete
#     num_cols_to_black = int(percent_to_remove / 100 * width)
#
#     # face coloanele negre stanga-dreatpta
#     frame_copy[:, :num_cols_to_black] = 0
#     frame_copy[:, -num_cols_to_black:] = 0
#
#     # gaseste coordonatele pixelilor albi
#     left_indices = np.argwhere(frame_copy[:, :width // 2] > 0)
#     right_indices = np.argwhere(frame_copy[:, width // 2:] > 0)
#
#     # extrage x si y
#     left_ys, left_xs = left_indices[:, 0], left_indices[:, 1]
#     right_ys, right_xs = right_indices[:, 0], right_indices[:, 1] + (width // 2)
#
#     return left_xs, left_ys, right_xs, right_ys
#
#
# def make_lines(in_frame):
#
#     left_ys, left_xs, right_ys, right_xs = find_street_markings_coordinates(in_frame)
#
#     left_line_coeffs = np.polyfit(left_xs, left_ys, deg=1)
#     right_line_coeffs = np.polyfit(right_xs, right_ys, deg=1)
#
#     #left_line_coeffs = np.polynomial.polynomial.polyfit(left_ys, left_xs, deg=1)
#     #right_line_coeffs = np.polynomial.polynomial.polyfit(right_ys, right_xs, deg=1)
#     #
#     left_top_y = 0
#     left_top_x = int((left_top_y - left_line_coeffs[1]) / left_line_coeffs[0])
#
#     left_bottom_y = new_height
#     left_bottom_x = int((left_bottom_y - left_line_coeffs[1]) / left_line_coeffs[0])
#
#     right_top_y = 0
#     right_top_x = int((right_top_y - right_line_coeffs[1]) / right_line_coeffs[0])
#
#     right_bottom_y = new_height
#     right_bottom_x = int((right_bottom_y - right_line_coeffs[1]) / right_line_coeffs[0])
#
#     min_x = -10 ** 15
#     max_x = 10 ** 8
#
#     # logica eliminare valori in interval
#     if not (min_x < left_top_x < max_x):
#         left_top_x = 0
#     if not (min_x < left_bottom_x < max_x):
#         left_bottom_x = 0
#     if not (min_x < right_top_x < max_x):
#         right_top_x = 0
#     if not (min_x < right_bottom_x < max_x):
#         right_bottom_x = 0
#
#     left_top = left_top_y, left_top_x
#     left_bottom = left_bottom_y, left_bottom_x
#     right_top = right_top_y + new_width // 2, right_top_x
#     right_bottom = right_bottom_y + new_width // 2, right_bottom_x
#
#
#     return left_top, left_bottom, right_top, right_bottom
#
#
# SCALE_PERCENT = 25
# THRESHOLD_VALUE = 170 # 127
# WHITE_COLOR = (255, 255, 255)
#
# cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')
#
# while True:
#     ret, frame = cam.read()
#
#     if ret is False:
#         break
#
#
#     new_width = int(frame.shape[1] * SCALE_PERCENT / 100)
#     new_height = int(frame.shape[0] * SCALE_PERCENT / 100)
#
#     frame = cv2.resize(frame, (new_width, new_height))
#     original = frame.copy()
#     image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     height, width = frame.shape[:2]
#     mask = np.zeros_like(image_gray, dtype=np.uint8)
#
#     # --------------------------------  creare trapez
#     trapez_bounds = limits(width, height)
#     cv2.fillConvexPoly(mask, trapez_bounds, WHITE_COLOR)
#     # -------------------------------- aplicare masca
#     masked_frame = cv2.bitwise_and(image_gray, image_gray, mask=mask)
#     # masked_frame = image_gray * mask * 255
#     # --------------------------------- Stretch
#     screen_bounds, stretched_frame = stretch(trapez_bounds, new_width, new_height)
#     # --------------------------------- BLUR
#     blurred_frame = cv2.blur(stretched_frame, ksize=(5, 5))
#     # ----------------------------- SOBEL
#     sobel_filter_uint8 = sobel(stretched_frame)
#     # ----------------------------- binary thing
#     # AICI SCHIMBI FRAME-UL PE CARE FACI
#     _, binary_frame = cv2.threshold(stretched_frame, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
#     # imaginea, valoarea, absolute white, tip de threshold
#
#     # # --------------------- LINIIIIIIIIII
#     left_top, left_bottom, right_top, right_bottom = make_lines(binary_frame)
#
#     cv2.line(binary_frame, left_top, left_bottom,
#              (200, 0, 0), 5)  # Linie stânga
#
#     cv2.line(binary_frame, right_top, right_bottom,
#              (100, 0, 0), 5)  # Linie dreapta
#     # cv2.line(binary_frame, (new_width // 2, 0), (new_width // 2, new_height), (255, 0, 0), 1)
#
#     # original_bounds = np.array([(new_width, 0), (0, 0), (0, new_height), (new_width, new_height)], dtype=np.float32)
#
#     # ------------------ FINAL?
#
#     final1 = np.zeros((new_height, new_width, 3), dtype=np.uint8)
#     cv2.line(final1, left_top, left_bottom, (255, 50, 50), 10)
#
#     trapez_bounds = np.float32(trapez_bounds)
#     screen_bounds = np.float32(screen_bounds)
#
#     matrix = cv2.getPerspectiveTransform(screen_bounds, trapez_bounds)
#     final_lines_left = cv2.warpPerspective(final1, matrix, (new_width, new_height))
#     #cv2.imshow('final1', final_lines_left)
#
#     left_ys1, left_xs1, right_ys1, right_xs1 = find_street_markings_coordinates(final_lines_left)
#     left_line_coords = left_ys1, left_xs1, right_ys1, right_xs1
#     #---------------------------------------------dreapta
#
#     final2 = np.zeros((new_height, new_width, 3), dtype=np.uint8)
#     #
#     cv2.line(final2, right_top, right_bottom, (50, 250, 50), 10)
#     #
#     matrix2 = cv2.getPerspectiveTransform(screen_bounds, trapez_bounds)
#     final_lines_right = cv2.warpPerspective(final2, matrix2, (new_width, new_height))
#     #cv2.imshow('final2', final_lines_right)
#     left_ys2, left_xs2, right_ys2, right_xs2 = find_street_markings_coordinates(final_lines_left)
#     right_line_coords = left_ys2, left_xs2, right_ys2, right_xs2
#
#
#     result_frame = original.copy()
#     frame_linii_color = final_lines_left + final_lines_right
#
#     cv2.imshow('combinat', frame_linii_color)
#     result_frame = result_frame + frame_linii_color * 255
#     result_frame = cv2.resize(result_frame, (500, 250))
#
#     cv2.imshow("Lane Detection with Colored Lines", result_frame)
#
#     cv2.imshow('Original', original)
#     cv2.imshow('Grayscale', image_gray)
#     cv2.imshow('Trapez', mask)
#     cv2.imshow('Road', masked_frame)
#     cv2.imshow('Top Down', stretched_frame)
#     cv2.imshow('Blurred Video', blurred_frame)
#     cv2.imshow('Sobel', sobel_filter_uint8)
#     cv2.imshow('Binary Image', binary_frame)
#
#     # close
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cam.release()
# cv2.destroyAllWindows()