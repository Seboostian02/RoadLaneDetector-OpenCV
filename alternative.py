# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture('Lane Detection Test Video-01.mp4')
#
#
# def process_frame(frame):
#     # Gri
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     # Canny
#     edges = cv2.Canny(blurred, 50, 150)
#
#     # Regiunea de interes
#
#     height, width = frame.shape[:2]
#
#     bottom_left = (int(width * 0.1), int(height * 1))
#     top_left = (int(width * 0.48), int(height * 0.77))
#     top_right = ((width // 2 + 50) , (height // 2 + 50) )
#     bottom_right = (int(width * 0.9), int(height * 1))
#
#     vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)
#     mask = np.zeros_like(edges)
#     cv2.fillPoly(mask, vertices, 255)
#     masked_edges = cv2.bitwise_and(edges, mask)
#
#     cv2.imshow('test', mask)
#     # linii magice
#     lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#     return lines
#
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     lines = process_frame(frame)
#
#     # Draw the detected lines on the frame
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
#
#     cv2.imshow('Lane Detection', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()