import cv2
import numpy as np


def limits(width_in, height_in):
    """
    :param width_in:
    :param height_in:
    :return: arr of limits  [upper_right, upper_left, lower_left, lower_right]
    """
    upper_left = (int(width_in * 0.40), int(height_in * 0.78))
    upper_right = (int(width_in * 0.60), int(height_in * 0.78))
    lower_left = (int(width_in * 0.1), int(height_in * 1))
    lower_right = (int(width_in * 0.9), int(height_in * 1))

    arr_of_limits = [upper_right, upper_left, lower_left, lower_right]

    return np.array(arr_of_limits, dtype=np.int32)


def stretch(in_trapez_bounds):
    """
    Stretch the frame
    :param in_trapez_bounds: margini trapez
    :param in_width: frame width
    :param in_height: frame height
    :return: screen_bounds and stretched frame
    """
    in_trapez_bounds = np.float32(in_trapez_bounds)
    out_screen_bounds = np.array([(width, 0), (0, 0), (0, height), (width, height)], dtype=np.float32)

    perspective_matrix = cv2.getPerspectiveTransform(in_trapez_bounds, out_screen_bounds)

    return out_screen_bounds, cv2.warpPerspective(masked_frame, perspective_matrix, (width, height))


def sobel(in_stretched_frame):
    sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    # kernelul pentru detectarea marginilor orizontale în imagine folosind filtrul Sobel.
    sobel_vertical = np.transpose(sobel_horizontal)
    # aceeasi chestie pentru orizontal


    # resized = resized.astype(np.float32)
    stretched_frame_float32 = np.float32(in_stretched_frame)

    horizontal_edges = cv2.filter2D(stretched_frame_float32, -1, sobel_horizontal)
    vertical_edges = cv2.filter2D(stretched_frame_float32, -1, sobel_vertical)

    sobel_filter = np.sqrt(np.square(horizontal_edges) + np.square(vertical_edges))

    return cv2.convertScaleAbs(sobel_filter)


def get_points_remove_noise(in_frame, percentage_to_remove = 5):
    """
    Optimizeaza frame-ul (sterge by default 5% stanga-dreapta)
    :param in_binary_frame: binary frame
    :return: puncte care urmeaza marcajele de pe strada
    left_ys, left_xs, right_ys, right_xs
    """
    frame_copy = in_frame.copy()

    frame_width = in_frame.shape[1]

    columns_to_remove = int(frame_width * (percentage_to_remove / 100))

    frame_copy[:, :columns_to_remove] = 0  # stergem primele coloane
    frame_copy[:, -columns_to_remove:] = 0  # stergem ultimele coloeane

    left_half = image_gray[:, : frame_width // 2]
    right_half = image_gray[:, frame_width // 2:]

    white_pixels_left = np.argwhere(left_half > 130)
    in_left_ys, in_left_xs = white_pixels_left[:, 0], white_pixels_left[:, 1]

    white_pixels_right = np.argwhere(right_half > 80)
    in_right_ys, in_right_xs = white_pixels_right[:, 0], white_pixels_right[:, 1]

    return in_left_ys, in_left_xs, in_right_ys, in_right_xs


def make_lines(in_frame):

    left_ys, left_xs, right_ys, right_xs = get_points_remove_noise(in_frame)

    left_pixels = np.column_stack((left_xs, left_ys))
    right_pixels = np.column_stack((right_xs, right_ys))

    # Calculare regresie liniara
    if len(left_pixels) > 0:
        left_line = np.polynomial.polynomial.polyfit(
            left_pixels[:, 1], left_pixels[:, 0], deg=1)

        left_top_y = 0
        left_bottom_y = frame.shape[0]  # height

        # rezolvare ecuatie y = m * x + b
        # left_line[1] = rata de schimbare a lui y fata de x (m)
        # left_line[0] = linia care traverseaza axa y(y = 0)
        left_top_x = int((0 - left_line[0]) / left_line[1])
        left_bottom_x = int((frame.shape[0] - left_line[0]) / left_line[1])

    if len(right_pixels) > 0:
        right_line = np.polynomial.polynomial.polyfit(
            right_pixels[:, 1], right_pixels[:, 0], deg=1)

        right_top_y = 0
        right_bottom_y = frame.shape[0]  # height

        right_top_x = int((0 - right_line[0]) / right_line[1])
        right_bottom_x = int((frame.shape[0] + right_line[0]) / right_line[1])

    min_x = -10 ** 8
    max_x = 10 ** 8

    # logica eliminare valori outliner
    if not (min_x < left_top_x < max_x):
        left_top_x = 0
    if not (min_x < left_bottom_x < max_x):
        left_bottom_x = 0
    if not (min_x < right_top_x < max_x):
        right_top_x = 0
    if not (min_x < right_bottom_x < max_x):
        right_bottom_x = 0

    # Verificare puncte proaste
    if abs(left_top_x) > 1e8:
        left_top_x = left_top_x if left_top_x != 0 else left_top_x
        #daca este diferit de 0 primeste valoarea anterioara
    if abs(left_bottom_x) > 1e8:
        left_bottom_x = left_bottom_x if left_bottom_x != 0 else left_bottom_x
    if abs(right_top_x) > 1e8:
        right_top_x = right_top_x if right_top_x != 0 else right_top_x
    if abs(right_bottom_x) > 1e8:
        right_bottom_x = right_bottom_x if right_bottom_x != 0 else right_bottom_x

    left_top = left_top_y, left_top_x
    left_bottom = left_bottom_y, left_bottom_x
    right_top = right_top_y + new_width // 2, right_top_x
    right_bottom = right_bottom_y + new_width // 2, right_bottom_x


    return left_top, left_bottom, right_top, right_bottom




SCALE_PERCENT = 25
THRESHOLD_VALUE = 190 # 127
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
    # masked_frame = image_gray * mask * 255
    # --------------------------------- Stretch
    screen_bounds, stretched_frame = stretch(trapez_bounds)
    # --------------------------------- BLUR
    blurred_frame = cv2.blur(stretched_frame, ksize=(5, 5))
    # ----------------------------- SOBEL
    sobel_filter_uint8 = sobel(stretched_frame)
    # ----------------------------- binary thing
    # AICI SCHIMBI FRAME-UL PE CARE FACI
    _, binary_frame = cv2.threshold(stretched_frame, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    # imaginea, valoarea, absolute white, tip de threshold

    # # --------------------- LINIIIIIIIIII
    left_top, left_bottom, right_top, right_bottom = make_lines(binary_frame)

    line_thickness = 5
    cv2.line(binary_frame, left_top, left_bottom,
             (200, 0, 0), line_thickness)  # Linie stânga

    cv2.line(binary_frame, right_top, right_bottom,
             (100, 0, 0), line_thickness)  # Linie dreapta

    middle_x = width // 2
    cv2.line(binary_frame, (middle_x, 0), (middle_x, height),
             (255, 0, 0), 1) # Linie mijloc

    # ------------------ FINAL

    final1 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.line(final1, left_top, left_bottom, (255, 50, 50), 10)

    trapez_bounds = np.float32(trapez_bounds)
    screen_bounds = np.float32(screen_bounds)

    matrix = cv2.getPerspectiveTransform(screen_bounds, trapez_bounds)
    final_lines_left = cv2.warpPerspective(final1, matrix, (width, height))

    left_ys1, left_xs1, right_ys1, right_xs1 = get_points_remove_noise(final_lines_left)
    left_line_coords = left_ys1, left_xs1, right_ys1, right_xs1

    # ---------------------------------------------dreapta
    final2 = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    cv2.line(final2, right_top, right_bottom, (50, 250, 50), 10)

    matrix2 = cv2.getPerspectiveTransform(screen_bounds, trapez_bounds)
    final_lines_right = cv2.warpPerspective(final2, matrix2, (width, height))

    left_ys2, left_xs2, right_ys2, right_xs2 = get_points_remove_noise(final_lines_left)
    right_line_coords = left_ys2, left_xs2, right_ys2, right_xs2


    result_frame = original.copy()
    frame_linii_color = final_lines_left + final_lines_right

    cv2.imshow('Combinat', frame_linii_color)

    # Creare masca linii
    non_black_pixels = np.any(frame_linii_color != [0, 0, 0], axis=-1)
    mask_lines = np.stack([non_black_pixels] , axis=-1)

    masked_lines_frame = np.where(mask_lines, frame_linii_color, result_frame)
    #result_frame = result_frame + frame_linii_color * 255
    alpha = 1  # Transparenta linii
    beta = 1.0 - alpha

    # Unim frame-urile
    result_frame = cv2.addWeighted(result_frame, beta, masked_lines_frame, alpha, 0)
    result_frame = cv2.resize(result_frame, (500, 250))


    cv2.imshow("Lane Detection with Colored Lines", result_frame)


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