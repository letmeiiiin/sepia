# ASR
from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import wave
import subprocess
import srt
import json
import datetime
import pyaudio
import wave
import time
import ctypes
from multiprocessing import Process, Value, Array

# Opencv
import cv2, time
import os.path

SetLogLevel(-1)

# Opencv
face_cascade = cv2.CascadeClassifier('../opencv/Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../opencv/Cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('../opencv/Cascades/haarcascade_smile.xml')

# Vosk / Pyaudio
model_path = "../asr/vosk/model"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

#manager = multiprocessing.Manager()

# Initialisation de la capture de la video
video_capture = cv2.VideoCapture(0)
frame_rate = 30
prev = 0

# Initialisation de Vosk
if not os.path.exists(model_path):
	print (f"Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in {model_path}")
	exit (1)

sample_rate = RATE
model = Model(model_path)
rec = KaldiRecognizer(model, sample_rate)
rec.SetWords(True)

# typecodes : https://docs.python.org/3/library/array.html#module-array
smile_detected = Value(ctypes.c_bool, False, lock=False)

# flow :
# asr() -> locate() -> smile_detection() -> take_pic()

# Définition des programmes de reconnaissances
def take_pic():
	cv2.imwrite("outputimage.jpg", frame2)
	print("took a picture !")

def detect_smile(frame):
	global smiles, smile_detected, debut
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces:
		roi_gray = gray[y:y + h, x:x + w]
		roi_color = frame[y:y + h, x:x + w]
		smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=20)
		debut = time.time()
		for (sx, sy, sw, sh) in smiles:
			cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
			if smiles.any():
				smile_detected.value = True
			else:
				smile_detected.value = False

	return frame

# Sound + Opencv based ?
def locate():
	print("moving ...")

# Opencv loop
def smile_detection():
	global frame2, video_capture, frame_rate, prev

	while True:
		time_elapsed = time.time() - prev

		_2, frame2 = video_capture.read()
		if time_elapsed > 1. / frame_rate:
			prev = time.time()

			test = detect_smile(frame2)
			cv2.imshow("Video2", test)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

# ASR loop
def asr():
	try : 

		# Initialisation du stream audio
		p = pyaudio.PyAudio()
		stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=8000)
		stream.start_stream()

		while True:
			data = stream.read(4000)
			if len(data) == 0:
				break
			if rec.AcceptWaveform(data):
				result = rec.FinalResult()
				#print(result)
				jres = json.loads(result)
				if "result" in jres:
					sentence = jres["text"]
					if "photo" in sentence:

						# Save data to wav
						wav = wave.open("photo_1.wav", 'wb')
						wav.setnchannels("1")
						wav.setsampwidth(p.get_sample_size(FORMAT))
						wav.setframerate(RATE)
						wav.writeframes(b''.join(data))
						wav.close()

						locate()
						
						while True:
							if smile_detected.value:
								print("smile detected !")
								take_pic()
								break

	except (KeyboardInterrupt, SystemExit):
		print("closed streams !")
		stream.stop_stream()
		stream.close()
		p.terminate()

try:
	smile_detection_p = Process(target=smile_detection)
	asr_p = Process(target=asr)

	smile_detection_p.start()
	asr_p.start()

	smile_detection_p.join()
	asr_p.join()

except (KeyboardInterrupt, SystemExit):
	print("Exiting...")

	video_capture.release()
	cv2.destroyAllWindows()

	smile_detection_p.terminate()
	asr_p.terminate()

	smile_detection_p.join()
	asr_p.join()

# RPI -> 1gb ram, Opencv ~ 300mb ram, Vosk ?
