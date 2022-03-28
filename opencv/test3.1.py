# Librairie importés
import threading

import cv2, time
import os.path

# On load les schémas
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')


# Définition des programmes de reconnaissances
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return frame


def detect_smile_and_picture(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    global smiles, debut
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            if smiles.any():
                t = threading.Timer(2.0, attendre_et_sauvegarder)
                t.start()
                break
    return frame


def attendre_et_sauvegarder():
    global filename
    filename = 'savedImage0.jpg'
    for i in range(0, 5):
        time.sleep(3)
        if os.path.exists(filename):
            filename = 'savedImage' + str(i + 1) + '.jpg'
            cv2.imwrite(filename, frame2)
            break
        else :
            cv2.imwrite(filename, frame2)


# On démarre la capture de la video
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_rate = 30
prev = 0

while True:
    time_elapsed = time.time() - prev

    _2, frame2 = video_capture.read()
    if time_elapsed > 1. / frame_rate:
        prev = time.time()

        test = detect_smile_and_picture(frame2)
        cv2.imshow("Video2", test)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
