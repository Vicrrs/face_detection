# bibliotecas
import cv2
import os


# Lendo o xml
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default1.xml')


def faceBlur(gray, frame):

    # Deteccao de face
    faces = cascade.detectMultiScale(gray, 1.1, 4)

    # Localizacao da face
    for (x, y, w, h) in faces:
        # ROI da face
        roi_frame = frame[y:y+h, x:x+w]

        # Filtro
        blur = cv2.GaussianBlur(roi_frame, (101, 101), 0)

        # Aplicacao do filtro
        frame[y:y+h, x:x+w] = blur

    return frame


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = faceBlur(gray, frame)
    cv2.imshow('Video', blur)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
