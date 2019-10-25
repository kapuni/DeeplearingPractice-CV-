import cv2
import numpy as np

image = cv2.imread("F:/python/image/test.jpg")
cv2.imshow("Image", image)

mask = np.zeros(image.shape[:2], dtype="uint8")

(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
cv2.rectangle(mask, (cX - 20, cY-400), (cX + 350, cY + 0), 255, -1)
cv2.imshow("Mask", mask)

masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)
