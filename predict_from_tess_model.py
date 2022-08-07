
import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder

Tess = "./data/TESS/TESS Toronto emotional speech set data/"

def main_f():
    tess_directory_list = os.listdir(Tess)
    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        directories = os.listdir(Tess + dir)
        for file in directories:
            part = file.split('.')[0]
            part = part.split('_')[2]
            if part=='ps':
                file_emotion.append('surprise')
            else:
                file_emotion.append(part)
            file_path.append(Tess + dir + '/' + file)
            
    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Tess_df = pd.concat([emotion_df, path_df], axis=1)
    # Tess_df.head()
    data_path = pd.concat([ Tess_df], axis = 0)
    data_path.to_csv("./data/data_path.csv",index=False)
    # data_path.head()
    
    path = np.array(data_path.Path)[1]
    # data, sample_rate = librosa.load(path)
    
    # X, Y = [], []
    # for path, emotion in zip(data_path.Path, data_path.Emotions):
    #     feature = get_features(path)
    #     for ele in feature:
    #         X.append(ele)
    #         # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
    #         Y.append(emotion)

    # len(X), len(Y), data_path.Path.shape

    # Features = pd.DataFrame(X)
    # Features['labels'] = Y
    # Features.to_csv('./data/features.csv', index=False)
    # Features.head()

    # X = Features.iloc[: ,:-1].values
    # Y = Features['labels'].values

    # encoder = OneHotEncoder()
    # Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

    return path

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data,sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result


def predict_emotion(file_name):
    labels = ['angry','calm','disgust','fear','happy','nuetral', 'sad', 'suprise']
    
    
    loaded_model = load_model("./trained_models/sound_model_original.h5", compile = True)
    scaler = StandardScaler()
    encoder = OneHotEncoder()
    
    path = main_f()

    file_location = f"./uploads/audio_records/{file_name}"

    res = get_features(file_location)
    scaler.fit_transform(res)
    x_res = scaler.transform(res)
    x_res = np.expand_dims(x_res, axis=2)
    
    pred_res = loaded_model.predict(x_res)
    pred_label = pred_res.argmax(axis=-1)
    
    print('pred_label',pred_label)
    
    final_result = ""
    if pred_label[2] == 0:
        final_result = "Angry"
    elif pred_label[2] == 1:
        final_result = "Calm"
    elif pred_label[2] == 2:
        final_result = "Disgust"
    elif pred_label[2] == 3:
        final_result = "Fear"
    elif pred_label[2] == 4:
        final_result = "Happy"
    elif pred_label[2] == 5:
        final_result = "Nuetral"
    elif pred_label[2] == 6:
        final_result = "Sad"
    elif pred_label[2] == 7:
        final_result = "Suprise"
    
    print('final_result',final_result)
    return final_result
        
    # y_pred = encoder.inverse_transform(pred_test)
    