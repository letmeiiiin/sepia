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
import audioop
#import RPi.GPIO as GPIO
from multiprocessing import Process, Value, Array
from array import array
from struct import pack
from pydub import AudioSegment
import numpy as np
import scipy as sp
import scipy.io.wavfile as spiow
import scipy.signal as spsig
import math
import matplotlib.pyplot as plt
from rpi_hardware_pwm import HardwarePWM
# Opencv
import cv2
import time
import os.path

SetLogLevel(-1)

# Opencv
face_cascade = cv2.CascadeClassifier(
    '../opencv/Cascades/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(
    '../opencv/Cascades/haarcascade_smile.xml')

# Vosk / Pyaudio
model_path = "../asr/vosk/model"
FORMAT = pyaudio.paInt16
CHANNELS = 1
VOSK_RATE = 16000

#manager = multiprocessing.Manager()

# Initialisation de la capture de la video
video_capture = cv2.VideoCapture(0)
frame_rate = 30
prev = 0

# Initialisation de Vosk
if not os.path.exists(model_path):
    print(
        f"Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in {model_path}")
    exit(1)

model = Model(model_path)
rec = KaldiRecognizer(model, VOSK_RATE)
rec.SetWords(True)

# typecodes : https://docs.python.org/3/library/array.html#module-array
smile_detected = Value(ctypes.c_bool, False, lock=False)

# flow :
# asr() -> locate() -> smile_detection() -> take_pic()

# Définition des programmes de reconnaissances


def detect_smile(frame):
    global smiles, smile_detected, debut
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.15, minNeighbors=20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy),
                          (sx + sw, sy + sh), (0, 0, 255), 2)
            if smiles.any():
                cv2.imwrite("outputimage.jpg", frame2)
                smile_detected.value = True
            else:
                smile_detected.value = False

    return frame

# Sound + Opencv based ?


def get_angle():
    #Fs = 16000
    Fs, x = spiow.read('mono_left.wav')
    Fss, y = spiow.read('mono_right.wav')

    x = np.array(x, dtype='int64')
    y = np.array(y, dtype='int64')

    dist = float(0.25)

    N = len(x)
    dur = N/Fs
    t = np.arange(0, dur, 1/Fs)

    Cxx = spsig.correlate(x, y, 'full')
    t_Cxx = np.arange(-dur+1/Fs, dur-1/Fs, 1/Fs)
    print(t_Cxx[Cxx.argmax()])
    rad = math.acos((340*t_Cxx[Cxx.argmax()])/dist)
    angles = np.degrees(rad)
    print(angles)
    return angles


def locate(channels):
    # record("output1.wav")
    # splitstereo(data_stereo_44100)
    l_channel = AudioSegment(
        channels[0], sample_width=2, channels=1, frame_rate=16000)
    r_channel = AudioSegment(
        channels[1], sample_width=2, channels=1, frame_rate=16000)
    l_channel.export("mono_left.wav", format="wav")
    r_channel.export("mono_right.wav", format="wav")
    angle = get_angle()
    print("getting angle")
    print(angle)
    return angle


# Fonction permettant le calcul de l'angle vers un format approprié pour le servomoteur SG90
def angle_to_percent(angle):
    if angle > 180 or angle < 0:
        return False

    start = 4
    end = 22
    ratio = (end - start)/360  #Calcul du ratio

    angle_as_percent = angle * ratio
    return start + angle_as_percent


# Opencv loop
def smile_detection():
    global frame2, video_capture, frame_rate, prev

    while True:
        time_elapsed = time.time() - prev

        _2, frame2 = video_capture.read()
        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            test = detect_smile(frame2)
            #cv2.imshow("Video2", test)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# ASR loop


def asr():
    try:

        # Initialisation du stream audio
        p = pyaudio.PyAudio()
        stream = p.open(input_device_index=2, format=pyaudio.paInt16,
                        channels=2, rate=44100, input=True, frames_per_buffer=10240)
        stream.start_stream()
        while True:
            data_stereo_44100 = stream.read(10240, exception_on_overflow=False)
            data_stereo_16000 = audioop.ratecv(
                data_stereo_44100, 2, 2, 44100, 16000, None)[0]
            data_mono_16000_l = audioop.tomono(data_stereo_16000, 2, 1, 0)
            data_mono_16000_r = audioop.tomono(data_stereo_16000, 2, 0, 1)
            if len(data_mono_16000_l) == 0:
                break
            if rec.AcceptWaveform(data_mono_16000_l):
                result_str = ""
                result = rec.FinalResult()
                print(result)
                for line in result.split('\n'):
                    if '  "text"' in line and line != '  "text" : ""':
                        result_str = line.split(":")[1][1:]
                #jres = json.loads(result)
                # print(jres)
                #a = False
                print(result_str)
                # print(type(data_mono_16000_r))
                #arr = np.frombuffer(data_mono_16000_r, dtype=np.int16, size=FRAME_LEN_B // 2)
                if "photo" in result_str:

                    pwm.start(100)
                    time.sleep(1)
                    # Init à 0°
                    pwm.change_duty_cycle(angle_to_percent(90))
                    time.sleep(1)
                    channels = (data_mono_16000_l, data_mono_16000_r)
                    angle1 = locate(channels)
                    if angle1 > 90 and angle1 < 180:
                        angle2 = 90-angle1
                        # Go à l'angle obtenu
                        pwm.change_duty_cycle(6+angle_to_percent(angle2))
                        time.sleep(1)
                    if angle1 > 0 and angle1 < 90:
                        angle2 = angle1
                        # Go à l'angle obtenu
                        pwm.change_duty_cycle(angle_to_percent(angle2))
                        time.sleep(1)
                    # pwm.stop()
                    while True:
                        if smile_detected.value:
                            print("smile detected !")
                            subprocess.run(["feh", "outputimage.jpg"])
                            raise SystemExit
                            break

    except (KeyboardInterrupt, SystemExit):
        print("closed streams !")
        stream.stop_stream()
        stream.close()
        p.terminate()
        exit()


# GPIO.setmode(GPIO.BOARD)  # Use Board numerotation mode
# GPIO.setwarnings(False)  # Disable warnings
# Use pin 12 for PWM signal
#pwm_gpio = 12
#frequence = 50
#GPIO.setup(pwm_gpio, GPIO.OUT)
pwm = HardwarePWM(pwm_channel=0, hz=60)

try:
    pwm.start(100)
    # remettre en position initiale
    pwm.change_duty_cycle(angle_to_percent(90))

    smile_detection_p = Process(target=smile_detection)
    asr_p = Process(target=asr)

    smile_detection_p.start()
    asr_p.start()

    smile_detection_p.join()
    asr_p.join()

except (KeyboardInterrupt, SystemExit):
    print("Exiting...")
    pwm.stop()

    video_capture.release()
    cv2.destroyAllWindows()

    smile_detection_p.terminate()
    asr_p.terminate()

    smile_detection_p.join()
    asr_p.join()


def splitstereo(stereoaudio):
    stereo_audio = AudioSegment.from_file(
        stereoaudio,
        format="wav")
    mono_audios = stereo_audio.split_to_mono()
    mono_left = mono_audios[0].export(
        "mono_left.wav",
        format="wav")
    mono_right = mono_audios[1].export(
        "mono_right.wav",
        format="wav")


def angle():
    Fs, x = spiow.read('mono_left.wav')
    Fss, y = spiow.read('mono_right.wav')

    x = np.array(x, dtype='int64')
    y = np.array(y, dtype='int64')
    print("entrez la distance")
    dist = float(input())

    N = len(x)
    dur = N/Fs
    t = np.arange(0, dur, 1/Fs)

    Cxx = spsig.correlate(x, y, 'full')
    t_Cxx = np.arange(-dur+1/Fs, dur-1/Fs, 1/Fs)
    print(t_Cxx[Cxx.argmax()])
    rad = math.acos((340*t_Cxx[Cxx.argmax()])/dist)
    a = 0
    coef = 1,4
    angle = np.degrees(rad)

    #angle = np.degrees(rad)
    print(angle)
    return angle


# RPI -> 1gb ram, Opencv ~ 300mb ram, Vosk ?
# Close GPIO & cleanup
# pwm.stop()
# GPIO.cleanup()
