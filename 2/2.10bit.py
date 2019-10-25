import cv2
import numpy as np

rectangle = np.zeros((100, 100), dtype="uint8")
cv2.rectangle(rectangle, (30, 30), (70, 70), 255, -1)
cv2.imshow("Rectangle", rectangle)

circle = np.zeros((100, 100), dtype="uint8")
cv2.circle(circle, (50, 50), 25, 255, -1)
cv2.imshow("Circle", circle)

bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)

bitwiseOr = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOr)

bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXor)

bitwiseNot = cv2.bitwise_not(rectangle, circle)
cv2.imshow("NOT", bitwiseNot)

cv2.waitKey(0)