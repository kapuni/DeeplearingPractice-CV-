import cv2
import numpy as np

image = cv2.imread("F:/python/image/test.jpg")

print(f"width: {image.shape[1]} pixels")
print(f"height: {image.shape[0]} pixels")
print(f"channels: {image.shape[2]}")

#2.4平移
N = np.float32([[1, 0, 25], [0, 1, 50]])
shifted_image = cv2.warpAffine(image, N, (image.shape[1], image.shape[0]))

#2.5旋转
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 135, 1.0)
Rotated_image = cv2.warpAffine(image, M, (w, h))

#2.6缩放
new_w, new_h = 500, 500
resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

#2.7翻转
# '0' 垂直翻转 , '1' 水平翻转, '-1' 垂直 + 水平
flipped_image = cv2.flip(image, -1)

#2.8裁剪
cropped_image = image[100:500, 500:1000]

# cv2.imshow("Image", image)
# cv2.imshow("shifted", shifted_image)
# cv2.imshow("Rotated", Rotated_image)
cv2.imshow("resized", resized_image)
cv2.imshow("flipped", flipped_image)
cv2.imshow("cropped", cropped_image)
cv2.waitKey(0)

# cv2.imwrite("F:/python/image/test1.jpg", image)

