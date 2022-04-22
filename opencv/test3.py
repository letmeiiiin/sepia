# Librairie importés
import cv2, time
import os.path

# On load les schémas
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')


# Définition des programmes de reconnaissances
def detect_faces(frame):                #Non utilisé dans le programme
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
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)             #Permet  la détection du visage, il faut changer 'scaleFactor' selon l'environnement augmenter si besoin de précision et diminuer dans le cas contraire.
    global smiles, debut
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.4, minNeighbors=20)     #Permet  la détection du sourire, il faut changer 'scaleFactor' selon l'environnement augmenter si besoin de précision et diminuer dans le cas contraire.
        debut = time.time()
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            while smiles.any():
                fin = time.time()
                if (fin - debut) > 2:
                    cv2.imwrite("outputimage.jpg", frame2)
                    break

    return frame


# On démarre la capture de la video
video_capture = cv2.VideoCapture(0)
frame_rate = 30
prev = 0

while True:
    time_elapsed = time.time() - prev

    _2, frame2 = video_capture.read()
    if time_elapsed > 1. / frame_rate:  #Limite le nombre de frame de capture en fonction de 'frame_rate'
        prev = time.time()

        test = detect_smile_and_picture(frame2)     #On test si un sourire est présent dans l'enregistrement en temps réel
        cv2.imshow("Video2", test)      #Retour de la caméra (à enlever si il y a besoin de performance)

        if os.path.exists('outputimage.jpg') :   #Stop le programmme si une image est déjà existante
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):    #Stop le programme en appuyant sur la touche 'q'
            break

video_capture.release()     #On arrete la capture video
cv2.destroyAllWindows()     #On détruit les fenêtres ouvertes de retour caméra
