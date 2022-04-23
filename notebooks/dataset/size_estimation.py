import os
from pydub import AudioSegment
from datetime import timedelta

dataset_duration=[]
for file in os.scandir("data"):
    seg = AudioSegment.from_file(file, format="wav")
    # len works fine on .wav but inaccurate on compressed formats
    dataset_duration.append(len(seg))

dataset_duration=sum(dataset_duration)
delta = str(timedelta(milliseconds=dataset_duration)).split(".")[0].split(" ")[0].split(":")
units=["h", "m", "s"]
dataset_duration_str=""
for d, u in zip(delta, units):
    dataset_duration_str += d + u + " "

print(f"dataset size: {dataset_duration_str}")