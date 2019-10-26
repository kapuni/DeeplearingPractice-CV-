import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
camera = cv2.VideoCapture(0)

while True:
    (success, image) = camera.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_rects = faceCascade.detectMultiScale(image, scaleFactor=1.1,
                                               minNeighbors=5,
                                               minSize=(30,30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)
    image_copy = image.copy()
    for (x, y, w, h) in faces_rects:
        img = cv2.rectangle(image_copy, (x, y),(x+w,y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + h]
        roi_color = img[y:y + h, x:x + h]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow("Face and Eyes", image_copy)

    if not success:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    camera.release()
    cv2.destroyAllWindows()
