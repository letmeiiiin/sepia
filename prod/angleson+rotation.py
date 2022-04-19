import RPi.GPIO as GPIO
import time
import numpy as np
import scipy as sp
import scipy.io.wavfile as spiow
import scipy.signal as spsig
import math
import matplotlib.pyplot as plt
from pydub import AudioSegment
import pyaudio
import wave
from array import array
from struct import pack

def record(outputfile):
  FORMAT = pyaudio.paInt16
  CHANNELS = 2
  RATE = 44100
  RECORDS_SECONDS=3
  CHUNK=1024

  p=pyaudio.PyAudio()
  stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
  print("recording...")

  frames=[]
  for i in range(0, int(RATE/CHUNK*RECORDS_SECONDS)):
    data=stream.read(CHUNK)
    frames.append(data)
  print("done recording")

  stream.stop_stream()
  stream.close()
  p.terminate()

  wf=wave.open(outputfile,'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(RATE)
  wf.writeframes(b''.join(frames))
  wf.close()
  

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
    Fs,x = spiow.read('mono_left.wav') 
    Fss,y=spiow.read('mono_right.wav')

    x = np.array(x,dtype='int64')
    y= np.array(y,dtype='int64')
    
    dist = float(0.25)

    N = len(x)
    dur = N/Fs
    t = np.arange(0,dur,1/Fs)

    Cxx = spsig.correlate(x,y,'full')
    t_Cxx = np.arange(-dur+1/Fs,dur-1/Fs,1/Fs)
    print(t_Cxx[Cxx.argmax()])
    rad=math.acos((340*t_Cxx[Cxx.argmax()])/dist)
    angles=np.degrees(rad)
    print(angles)
    return angles

def locate():
    record("output1.wav")
    splitstereo('output1.wav')
    angle1=angle()
    print("getting angle")
    return angle1


#Set function to calculate percent from angle
def angle_to_percent (angle) :
    if angle > 180 or angle < 0 :
        return False

    start = 4
    end = 16
    ratio = 2*(end - start)/360 #Calcul ratio from angle to percent

    angle_as_percent = angle * ratio

    return start + angle_as_percent

GPIO.setmode(GPIO.BOARD) #Use Board numerotation mode
GPIO.setwarnings(False) #Disable warnings

#Use pin 12 for PWM signal
pwm_gpio = 12
frequence = 50
GPIO.setup(pwm_gpio, GPIO.OUT)
pwm = GPIO.PWM(pwm_gpio, frequence)

pwm.start(0)
time.sleep(1)
#Init à 0°
pwm.start(angle_to_percent(90))
time.sleep(1)
angle1 = locate()
if angle1 > 90 and angle1 < 180 :
    angle2 = angle1
if angle1 > 0 and angle1 < 90 :
    angle2 = angle1
#Go à l'angle obtenu
pwm.ChangeDutyCycle(angle_to_percent(angle2)) 
time.sleep(1)

#remettre en position initiale
#pwm.start(angle_to_percent(90))
#time.sleep(1)

#Close GPIO & cleanup
pwm.stop()
GPIO.cleanup()