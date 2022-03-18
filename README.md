# sepia

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
Le code ne tourne pas entièrement sur binder (modèle trop lourd), ne fonctionne pas pour l'instant.

Utilisation :
- Le modèle Vosk https://alphacephei.com/vosk/models/vosk-model-small-fr-pguyot-0.3.zip a été extrait dans le dossier "model", pas besoin de le re-extraire.
- Installer ffmpeg https://ffmpeg.org/download.html
- git clone https://github.com/letmeiiiin/sepia.git
- cd sepia
- Créer un environement virtuel sous python 3.9
- pip install -r requirements.txt
- Lancer le notebook "dataset.ipynb"

Détails :
- Prends ~ 40s (éxecution complète du notebook) sur un i7-9750H avec le modèle vosk-model-small-fr-pguyot-0.3.zip
- Dans la vidéo d'exemple, 23 mots sur 26 sont transcrits correctement.