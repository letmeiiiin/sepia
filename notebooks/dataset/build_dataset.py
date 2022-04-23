import pandas as pd
from youtubesearchpython import VideosSearch
import time
import base64
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import *
import os
import sys
from yt_dlp import YoutubeDL
from pydub import AudioSegment, silence
from pydub.playback import play
import subprocess
import numpy as np
import soundfile as sf
import os.path
from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
import srt
import json
import datetime

pd.set_option('display.width', 1000)
sample_rate = 16000

vid_target = 1000
to_download = []
to_download_len = []
search_words = ["photo", "photos", "photographie", "photographier", "photographe"]
keywords = ["sur", "dans", "avec", "sans", "comment", "pourquoi", "la", "le", "les", "de", "des", "en", "filtre", "montage", "retouche", "apareil", "paysage", "portrait", "plan", "composition", "belle", "jolie", "astuce", "conseil", "cool", "facile", "super", "capteur", "collage", "papier", "galerie", "format", "pellicule", "argentique", "couleur", "artistique"]
search_terms=[]

for word in search_words:
	for keyword in keywords:
		search_terms.append(keyword + " " + word)

def fetch_encode_wav(url, filename):
	# Fetch best quality audio, re-encode to mono, 256kbps 16bit wav
	# to check file bitrate : ffprobe audio.wav

	filename = "audio/" + filename + ".wav"
	print(filename)

	video_info = YoutubeDL().extract_info(
		url = url,download=False
	)

	options={
		'format':'bestaudio/best',
		'postprocessors': [{
		'key': 'FFmpegExtractAudio',
		'preferredcodec': 'wav',
		}],
		'postprocessor_args': [
			'-ar', '16000',
			'-ac', '1',
		],
		'prefer_ffmpeg': True,
		'keepvideo':False,
		'outtmpl':filename,
	}

	with YoutubeDL(options) as ydl:
		ydl.download([video_info['webpage_url']])

	print("Download complete !")

def cut_merge(cuts, audio_in, audio_out):
	# Takes a list of tuples as cuts [(start, end)...]

	cuts_old = len(cuts) 
	for i in range(len(cuts)):

		if i < len(cuts)-1 and cuts[i][1] == cuts[i+1][0]:
			cuts[i+1] = (cuts[i][0], cuts[i+1][1])
			cuts.pop(i)

	print(f"total cuts : {len(cuts)}, removed {cuts_old - len(cuts)} useless cut(s)") 

	# Opening file and extracting segment
	segment = AudioSegment.from_file(audio_in, format="wav", frame_rate=sample_rate, channels=1)
	clips = []

	for cut in cuts:
		# cut_start, cut_end should be expressed in ms
		clips.append(segment[cut[0]*1000:cut[1]*1000])

	merged_clips = sum(clips)
	merged_clips.export(audio_out, format="wav", parameters=["-ar", str(sample_rate), "-ac", "1"])

def transcribe(words_per_line):
	results = []
	subs = []
	while True:
		data = process.stdout.read(4000)
		if len(data) == 0:
			break
		if rec.AcceptWaveform(data):
			results.append(rec.Result())
	results.append(rec.FinalResult())

	for i, res in enumerate(results):
		jres = json.loads(res)
		if not 'result' in jres:
			continue
		words = jres['result']
		for j in range(0, len(words), words_per_line):
			line = words[j : j + words_per_line]
			s = srt.Subtitle(index=len(subs), 
			content=" ".join([l['word'] for l in line]),
			start=line[0]['start'],
			end=line[-1]['end'])
			subs.append(s)
	return subs

search_terms_len=len(search_terms)
for i_term, term in enumerate(search_terms):
	print(f"term '{term}': {i_term}/{search_terms_len}")
	to_download = []
	to_download_len = []
	search = VideosSearch(term, language='fr', region='FR')
	vid_result = search.result()['result']

	while len(to_download) <= vid_target:

		if len(to_download_len) >= 3:
			same_elements = all(element == to_download_len[0] for element in to_download_len)
			if same_elements:
				break
			to_download_len = []

		for val in vid_result:
			to_download.append(val["link"])

		vid_result = search.next()
		vid_result = search.result()['result']

		to_download_len.append(len(to_download))
		
		print(f"{len(to_download)} urls to go through")
		time.sleep(0.2)

	if not os.path.exists("dataset.csv"):
		cols = ['base64_url', 'labels']
		dataset_df = pd.DataFrame(columns=cols)
		dataset_df.to_csv('dataset.csv', columns=cols, index=False, header=True)

	try:
		dataset_df = pd.read_csv('dataset.csv')
	except pd.errors.EmptyDataError:
		pass

	to_download_len=len(to_download)
	for i_url, url in enumerate(to_download):
		print(f"url: {i_url}/{to_download_len}")

		base64_url_filename = str(base64.urlsafe_b64encode(url.encode("utf-8")), "utf-8")

		if base64_url_filename not in dataset_df.values:

			vid_id = url.partition("/watch?v=")[2]

			try:
				#retrieve auto-generated transcripts
				transcript_list = YouTubeTranscriptApi.list_transcripts(vid_id) 
				transcript = transcript_list.find_generated_transcript(['fr', 'fr-FR']) 
				if transcript == "":
					transcript = transcript_list.find_manually_created_transcript(['fr', 'fr-FR']) 
				if transcript != "":
					df = pd.DataFrame(transcript.fetch())
					#df.to_csv("transcript.csv", index=False)

					#df = pd.read_csv("transcript.csv")
					df.insert(2, 'stop', df['start'] + df['duration'], False)

					# Select rows with matching word regex
					word = "photo"
					df2 = df[df['text'].str.contains(r'\b'+word+r'\b', regex=True)]
					print(f"{df2.shape[0]} occurrence(s) of word : {word}")

					if df2.shape[0] > 1:

						fetch_encode_wav(url, base64_url_filename)

						#print(df2)
						# Make audio, generate SRT

						cut_merge([(start_time,stop_time) for start_time,stop_time in zip(df2['start'], df2["stop"])], "audio/" + base64_url_filename + ".wav", "audio/" + base64_url_filename + "_clip" + ".wav")
						os.remove("audio/" + base64_url_filename + ".wav")

						SetLogLevel(-1)
						model_path = "../../asr/vosk/model"

						if not os.path.exists(model_path):
							print (f"Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in {model_path}")
							exit (1)

						model = Model(model_path)
						rec = KaldiRecognizer(model, sample_rate)
						rec.SetWords(True)
						audio_clip = "audio/" + base64_url_filename + "_clip" + ".wav"

						process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
													audio_clip,
													'-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'],
													stdout=subprocess.PIPE)

						print(f"term '{term}': {i_term}/{search_terms_len}")
						print(f"url: {i_url}/{to_download_len}")

						# 3 tries, skip if still fails
						tries = 0
						labels = []
						while True:
							print("transcribing...")
							words = transcribe(1)
							for word in words:
								if word.content == "photo":
									labels.append((word.start, word.end))
							if labels or tries >= 2:
								break
							tries += 1

						if labels:
							cut_merge(labels, "audio/" + base64_url_filename + "_clip" + ".wav", "data/" + base64_url_filename + ".wav")
							print(f"{len(labels)} words transcribed !")

						if base64_url_filename not in dataset_df.values:
							dataset_df = pd.concat([dataset_df, pd.DataFrame.from_records([{'base64_url' : base64_url_filename, 'labels' : labels}])])
							dataset_df.to_csv('dataset.csv', mode='w', index=False, header=True)
							print(dataset_df.size)
							print(dataset_df)

						for file in os.scandir("audio"):
							os.remove(file.path)

					else:
						print("skipping url...")

			except (TranscriptsDisabled, NoTranscriptFound):
				print(f"No transcript found for: {url}")