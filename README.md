# sepia

# /notebooks/dataset

But :
Créer un dataset de mots (ici "photo"), de différents orateurs à partir du son de vidéos youtube.
Vidéo utilisée en exemple dans le notebook : https://www.youtube.com/watch?v=wDAmezoNHJY

Fonctionnement :
- Chope les transcriptions manuelles de vidéos youtube
- Stocke les données dans une dataframe
- Télécharge l'audio de la video, puis convertissement en .wav (mono)
- Garde seulements les passages où le mot "photo" apparaît
- Enlève les silences (inutile ?)
- Utilise vosk: https://github.com/alphacep/vosk-api pour transcrire l'audio mot par mot.
- Stocke les timestamps de chaque mot dans une liste de tuples
- Joue le fichier audio final

Détails :
- Moins bonne transcription quand le fichier sans silences est utilisé,
  donc le fichier normal est utilisé.
- Prends ~ 50s sur un i7-9750H
- Dans la vidéo d'exemple, entre 18 et 21 mots sur 26 sont transcrits (plus souvent 21 que 18).
  
Preview binder :
Le code ne tourne pas entièrement sur binder (modèle trop lourd).

Usage :
- Télécharger un modèle vosk FR, depuis https://alphacephei.com/vosk/modelsex
  ex : https://alphacephei.com/vosk/models/vosk-model-fr-0.6-linto-2.2.0.zip
- Extraire le contenu du modèle dans le dossier "model" (am, conf, graph, ivector, rescore...)
- Installer ffmpeg https://ffmpeg.org/download.html
- git clone https://github.com/letmeiiiin/sepia.git
- cd sepia
- Créer un environement virtuel sous python 3.9, installer les dépendances (fichier environment.yml pour conda dans le repo
  ou requirements.txt pour pip)
- Jouer avec le notebook "dataset.ipynb"
