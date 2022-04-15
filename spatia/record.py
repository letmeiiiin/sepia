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
  for i in range(0,int(RATE/CHUNK*RECORDS_SECONDS)):
    data=stream.read(CHUNK)
    frames.append(data)
  print(" done recording")

  stream.stop_stream()
  stream.close()
  p.terminate()

  wf=wave.open(outputfile,'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(RATE)
  wf.writeframes(b''.join(frames))
  wf.close()

record('output1.wav')