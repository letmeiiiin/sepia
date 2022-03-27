# Mise en route
- installation de ffmpeg et de portaudio : sudo apt update && sudo apt install portaudio19-dev ffmpeg
- création d'un environement virtuel python 3.9
- Si pyenv est utilisé, il faut créer l'environement avec : PYTHON_CONFIGURE_OPTS "--enable-shared"
- git clone https://github.com/letmeiiiin/sepia.git && cd sepia
- pip install -r requirements.txt

# /notebooks/dataset

But :
Créer un dataset de mots (ici "photo"), de différents orateurs à partir du son de vidéos youtube.
Vidéo utilisée en exemple dans le notebook : https://www.youtube.com/watch?v=wDAmezoNHJY

Fonctionnement :
- Télécharge les transcriptions manuelles de vidéos youtube
- Stocke les données dans une dataframe
- Télécharge l'audio de la video, puis convertissement en .wav (mono)
- Garde seulement les passages où le mot "photo" apparaît
- Enlève les silences (inutile ?)
- Utilise vosk: https://github.com/alphacep/vosk-api pour transcrire l'audio mot par mot.
- Stocke les timestamps de chaque mot dans une liste de tuples
- Joue le fichier audio final
  
Preview binder : https://mybinder.org/v2/gh/letmeiiiin/sepia/binder-dataset?labpath=dataset.ipynb
ne fonctionne pas pour l'instant.

Utilisation :
- Le modèle Vosk https://alphacephei.com/vosk/models/vosk-model-small-fr-pguyot-0.3.zip a été extrait dans le dossier "model", pas besoin de le re-extraire.
- Lancer le notebook "dataset.ipynb"

Détails :
- Prends ~ 40s (éxecution complète du notebook) sur un i7-9750H avec le modèle vosk-model-small-fr-pguyot-0.3.zip
- Dans la vidéo d'exemple, 23 mots sur 26 sont transcrits correctement.

# /notebooks/wav2vec2
But :
Utiliser un modèle wav2vec2 afin d'identifier en temps réel les mots prononcés par un orateur et réagir au mot clé "photo".

Utilisation :
- Configurer un microphone comme microphone par défaut
- Lancer le notebook "wav2vec2_live_inference.ipynb"

# /notebooks/vosk
But :
Utiliser Vosk API afin d'identifier en temps réel les mots prononcés par un orateur et réagir au mot clé "photo".
