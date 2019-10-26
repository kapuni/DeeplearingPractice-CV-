import cv2
import numpy as np

image = cv2.imread("F:/python/image/test.jpg")
cv2.imshow("Original", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
#
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
#
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*A*B", lab)
cv2.waitKey(0)
