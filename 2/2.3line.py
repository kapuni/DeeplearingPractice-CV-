import cv2
import numpy as np

canvas = np.zeros((300, 300, 3), dtype='uint8')

for _ in range(0, 25):
    radius = np.random.randint(5, 200)
    color = np.random.randint(0, 256, size=(3, )).tolist()
    pt = np.random.randint(0, 200, size=(2, ))

    cv2.circle(canvas, tuple(pt), radius, color, -1)

cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
