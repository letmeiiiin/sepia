from pydub import AudioSegment

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
	
splitstereo('output1.wav')