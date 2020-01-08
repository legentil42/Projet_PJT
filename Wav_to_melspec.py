#@title MEL SPECTROGRAM
#!rm -rf /content/melspectrogram/ #supprime le dossier melspectrogram si problèmes
!pip install librosa # installation de la librairie librosa
!git clone https://github.com/Jakobovski/free-spoken-digit-dataset #téléchargement du jeu de données
!mkdir /content/melspectrogram #création du dossier qui contiendra les mel-spectrogrammes

import glob # librairie spécialisée dans la recherche de chemin
list_of_files = glob.glob("free-spoken-digit-dataset/recordings/*") #permet d'obtenir le chemin relatif de tous les fichiers du dossier recordings
print(list_of_files[:2])

n_files = len(list_of_files) # nombres de fichiers audios
print(n_files)

import librosa
y, sr = librosa.load(list_of_files[3], sr= 8000) #y est le signal sous forme d'array numpy, sr est le taux d'échantillonage (vaut 8khz  d'après les indications github)
print(y, sr)

import numpy as np
sp = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128) # calcul du melspectrogramme
s_dB = librosa.power_to_db(sp, ref=np.max) #conversion de l'amplitude en dB pour une meilleure vi

print(type(s_dB))

from IPython.display import Audio
Audio(y,rate=sr) # permet de lire un fichier audio sous forme d'array numpy dans un notebook jupyter ou dans colab

import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def plot_spec(S_dB):
  plt.figure(figsize=(10, 4))
  librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr)
  plt.colorbar(format='%+2.0f dB')
  plt.title('Mel-frequency spectrogram')
  plt.tight_layout()
  plt.show()
plot_spec(s_dB)


n_mels=128
sr = 8000


def compute_spectrogram(path):
    "Calcule le melspectrogramme du fichier audio situé à path dans le repertoire de fichiers"
    file_name = os.path.basename(path) 
    s , _ = librosa.load(path, sr= sr) # sr est le taux d'échantillonnage du signal.
    
    melspec = librosa.feature.melspectrogram(y=s, sr=sr, n_mels=n_mels)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    melspec_db_image = melspec_db.reshape(melspec_db.shape[0],melspec_db.shape[1],1) # on rajoute la dimension 1 à la fin pour forcer la représentation en tant qu'image (hauteur, largeur, nombre de couleurs ici égale à 1)
    return (path,melspec_db_image)

def get_duration(spec):
  "Renvoie la dimension correspondant à l'axe temporel d'un melspectrogramme"
  return spec.shape[1]

def get_longest_duration(list_path_spec):
  "Renvoie la dimension temporelle du melspectrogramme le plus long"
  list_time = [get_duration(spec) for path, spec in list_path_spec ]
  return max(list_time)

def save_melspec(path_melspec): 
    """
    Permet d'enregistrer les spectrogrammes dans le dossier melspectrogram
    """
    path, melspec = path_melspec
    file_name = os.path.basename(path)
    melspec_path = os.path.join("/content/melspectrogram", os.path.splitext(file_name)[0])
    np.save(melspec_path, melspec)
    #print("{} saved".format(melspec_path))


import os
from tqdm.autonotebook import tqdm # bar de remplissage pour visualiser l'évolution du calcul et de l'enregistrement des spectrogrammes
import sys
import tensorflow as tf
n=n_files



print("Conversion des fichiers audio en melspectrogrammes")
list_spec = []
for path, spec in tqdm(map(compute_spectrogram, list_of_files[:n]), total=n):
   list_spec.append((path, spec))
   

max_time = get_longest_duration(list_spec)
print("La plus longue dimension temporelle est {}".format(max_time))
def spec_padding(path_data):
  path, spec = path_data
  min_spec = np.min(spec)
  spec_padded = np.pad(spec, ((0,0),(0,max_time-spec.shape[1]),(0,0)),"constant",constant_values=((min_spec,min_spec),(min_spec,min_spec),(min_spec,min_spec)))
  return (path, spec_padded)



print("Traitement des melspectrogrammes pour qu'ils aient la même durée")
list_padded_spec = []
for path, spec_padded in tqdm(map(spec_padding, list_spec[:n]), total=n):
  list_padded_spec.append((path,spec_padded))
  


print("Traitement des melspectrogrammes pour qu'ils aient la même durée")
for spec in tqdm(map(save_melspec, list_padded_spec[:n]), total=n):
  pass
