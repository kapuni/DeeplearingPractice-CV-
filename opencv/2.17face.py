import cv2

image = cv2.imread('F:/python/image/facewang1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_rect = faceCascade.detectMultiScale(image,
                                          scaleFactor=1.05,
                                          minNeighbors=5, minSize=(30,30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
for (x, y, w, h) in faces_rect:
    img = cv2.rectangle(image, (x, y), (x + w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+h]
    roi_color = img[y:y+h, x:x+h]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
