import pyaudio
import wave

from array import array
from struct import pack

def record(outputfile):
  FORMAT = pyaudio.paInt16
  CHANNELS = 2          #son stéréo qui sera après découpé en 2 via le programme split.py
  RATE = 44100          #freq d'échantillonage
  RECORDS_SECONDS=3   #tps en seondes du record
  CHUNK=1024

  p=pyaudio.PyAudio()
  stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
  print("recording...")

  frames=[]
  for i in range(0,int(RATE/CHUNK*RECORDS_SECONDS)):
    data=stream.read(CHUNK)
    frames.append(data)
  print(" done recording")

  stream.stop_stream()
  stream.close()
  p.terminate()

  wf=wave.open(outputfile,'wb')             #enregistrement du record sous le bo format
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(RATE)
  wf.writeframes(b''.join(frames))
  wf.close()

record('output1.wav')