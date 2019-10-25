import cv2
import numpy as np

image = cv2.imread("F:/python/image/test.jpg")
(B, G, R) = cv2.split(image)
merged = cv2.merge([B, G, R])

cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)

cv2.imshow("Merged", merged)
cv2.waitKey(0)
