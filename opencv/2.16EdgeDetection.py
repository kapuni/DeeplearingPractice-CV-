import numpy as np
import cv2

image = cv2.imread("cat1.jpg")
cv2.imshow("image", image)

blured = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blured = cv2.GaussianBlur(blured, (5, 5), 0)
cv2.imshow("Blured", blured)

canny = cv2.Canny(blured, 30, 150)
cv2.imshow("Cannt", canny)
cv2.waitKey(0)