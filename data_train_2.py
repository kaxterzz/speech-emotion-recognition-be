import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# librosa is a Python library for analyzing audio and music.
# It can be used to extract the data from the audio files we will see it later
import librosa 
import librosa.display

# to play the audio files
from IPython.display import Audio
plt.style.use('seaborn-white')

DATA_FRAMES = True
fem_path = '/data/Female_features/Female_features.csv'
mal_path = '/data/Male_features/Male_features.csv'

TESS = "/data/TESS/TESS Toronto emotional speech set data/"
RAV = "/data/RAVDESS/audio_speech_actors_01-24/"
SAVEE = "/data/surrey-audiovisual-expressed-emotion-savee/ALL/"
CREMA = "/data/cremad/AudioWAV/"