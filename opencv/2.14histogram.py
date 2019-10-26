import cv2
from matplotlib import pyplot as plt

image = cv2.imread("F:/python/image/test.jpg")
# cv2.imshow("image", image)
# cv2.waitKey(0)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

plt.figure()

p1 = plt.subplot(121)
p2 = plt.subplot(122)

p1.plot(hist)

chans = cv2.split(image)
colors = ("b", "g", "r")

#Color Histogram
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    p2.plot(hist, color=color)

plt.show()
