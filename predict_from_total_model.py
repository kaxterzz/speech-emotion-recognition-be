import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, AveragePooling1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

# librosa is a Python library for analyzing audio and music.
# It can be used to extract the data from the audio files we will see it later
import librosa 
import librosa.display

# to play the audio files
from IPython.display import Audio
plt.style.use('seaborn-white')

DATA_FRAMES = True
fem_path = './data/Female_features.csv'
mal_path = './data/Male_features.csv'

TESS = "./data/TESS/tess toronto emotional speech set data/"
RAV = "./data/RAVDESS/audio_speech_actors_01-24/"
SAVEE = "./data/SAVEE/ALL/"
CREMA = "./data/CREMA-D/AudioWAV/"

def main():
    # Get the data location for SAVEE
    dir_list = os.listdir(SAVEE)

    # parse the filename to get the emotions
    emotion=[]
    path = []
    for i in dir_list:
        if i[-8:-6]=='_a':
            emotion.append('angry')
        elif i[-8:-6]=='_d':
            emotion.append('disgust')
        elif i[-8:-6]=='_f':
            emotion.append('fear')
        elif i[-8:-6]=='_h':
            emotion.append('happy')
        elif i[-8:-6]=='_n':
            emotion.append('neutral')
        elif i[-8:-6]=='sa':
            emotion.append('sad')
        elif i[-8:-6]=='su':
            emotion.append('surprise')
        else:
            emotion.append('unknown') 
        path.append(SAVEE + i)

    # Now check out the label count distribution 
    SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])
    SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path, columns = ['path'])], axis = 1)
    # print('SAVEE dataset')
    # SAVEE_df.head()

    # Get the data location for TESS
    path = []
    emotion = []
    dir_list = os.listdir(TESS)

    for i in dir_list:
        fname = os.listdir(TESS + i)   
        for f in fname:
            if i == 'OAF_angry' or i == 'YAF_angry':
                emotion.append('angry')
            elif i == 'OAF_disgust' or i == 'YAF_disgust':
                emotion.append('disgust')
            elif i == 'OAF_Fear' or i == 'YAF_fear':
                emotion.append('fear')
            elif i == 'OAF_happy' or i == 'YAF_happy':
                emotion.append('happy')
            elif i == 'OAF_neutral' or i == 'YAF_neutral':
                emotion.append('neutral')                                
            elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':
                emotion.append('surprise')               
            elif i == 'OAF_Sad' or i == 'YAF_sad':
                emotion.append('sad')
            else:
                emotion.append('Unknown')
            path.append(TESS + i + "/" + f)

    TESS_df = pd.DataFrame(emotion, columns = ['labels'])
    #TESS_df['source'] = 'TESS'
    TESS_df = pd.concat([TESS_df,pd.DataFrame(path, columns = ['path'])],axis=1)
    # print('TESS dataset')
    # TESS_df.head()

    # Importing datas from RAVDESS
    dir = os.listdir(RAV)

    males = []
    females = [] 
            
    for actor in dir:
        
        files = os.listdir(RAV + actor)
            
        for file in files: 
            part = file.split('.')[0]
            part = part.split("-")           
                
            temp = int(part[6])        
                    
            if part[2] == '01':
                emotion = 'neutral'
            elif part[2] == '02':
                emotion = 'calm'
            elif part[2] == '03':
                emotion = 'happy'
            elif part[2] == '04':
                emotion = 'sad'
            elif part[2] == '05':
                emotion = 'angry'
            elif part[2] == '06':
                emotion = 'fear'
            elif part[2] == '07':
                emotion = 'disgust'
            elif part[2] == '08':
                emotion = 'surprise'
            else:
                emotion = 'unknown'
                
            if temp%2 == 0:
                path = (RAV + actor + '/' + file)
                #emotion = 'female_'+emotion
                females.append([emotion, path]) 
            else:
                path = (RAV + actor + '/' + file)
                #emotion = 'male_'+emotion
                males.append([emotion, path])   
        
    
    RavFemales_df = pd.DataFrame(females)
    RavFemales_df.columns = ['labels', 'path']

    RavMales_df = pd.DataFrame(males)
    RavMales_df.columns = ['labels', 'path']

    # print('RAVDESS datasets')
    # RavFemales_df.head()

    files = os.listdir(CREMA)

    female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,
            1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]
    males = []
    females = []

    for file in files: 
        part = file.split('_')   
        
        if part[2] == 'SAD':
            emotion = 'sad'
        elif part[2] == 'ANG':
            emotion = 'angry'
        elif part[2] == 'DIS':
            emotion = 'disgust'
        elif part[2] == 'FEA':
            emotion = 'fear'
        elif part[2] == 'HAP':
            emotion = 'happy'
        elif part[2] == 'NEU':
            emotion = 'neutral'  
        else:
            emotion = 'unknown'
            
        if int(part[0]) in female:
            path = (CREMA + '/' + file)
            #emotion = 'female_'+emotion
            females.append([emotion, path]) 
        else:
            path = (CREMA + '/' + file)
            #emotion = 'male_'+emotion
            males.append([emotion, path])   
        
    CremaFemales_df = pd.DataFrame(females)
    CremaFemales_df.columns = ['labels', 'path']

    CremaMales_df = pd.DataFrame(males)
    CremaMales_df.columns = ['labels', 'path']
        
    # print('CREMA datasets')
    # CremaFemales_df.head()

    # Now lets merge all the dataframe
    Males = pd.concat([SAVEE_df, RavMales_df, CremaMales_df], axis = 0)
    Males.to_csv("males_emotions_df.csv", index = False)

    Females = pd.concat([TESS_df, RavFemales_df, CremaFemales_df], axis = 0)
    Females.to_csv("females_emotions_df.csv", index = False)

    order = ['angry','calm','disgust','fear','happy','neutral','sad','surprise']

    # fig = plt.figure(figsize=(17, 5))

    # fig.add_subplot(121)
    # plt.title('Count of Females Emotions', size=16)
    # sns.countplot(Females.labels, order = order)
    # plt.ylabel('Count', size=12)
    # plt.xlabel('Emotions', size=12)
    # sns.despine(top=True, right=True, left=False, bottom=False)

    # fig.add_subplot(122)
    # plt.title('Count of Males Emotions', size=16)
    # sns.countplot(Males.labels, order = order)
    # plt.ylabel('Count', size=12)
    # plt.xlabel('Emotions', size=12)
    # sns.despine(top=True, right=True, left=False, bottom=False)

    # plt.show()


def noise(data):
    noise_amp = 0.04*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.70):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.8):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def higher_speed(data, speed_factor = 1.25):
    return librosa.effects.time_stretch(data, speed_factor)

def lower_speed(data, speed_factor = 0.75):
    return librosa.effects.time_stretch(data, speed_factor)

def extract_features(data):
    
    result = np.array([])
    
    #mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=42) #42 mfcc so we get frames of ~60 ms
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    result = np.array(mfccs_processed)
     
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    # data, sample_rate = librosa.load(path, duration=3, offset=0.5, remains_type='kaiser_fast') 
    data, sample_rate = librosa.load(path, duration=3, offset=0.5) 

    #without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
    #noised
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically
    
    #stretched
    stretch_data = stretch(data)
    res3 = extract_features(stretch_data)
    result = np.vstack((result, res3))
    
    #shifted
    shift_data = shift(data)
    res4 = extract_features(shift_data)
    result = np.vstack((result, res4))
    
    #pitched
    pitch_data = pitch(data, sample_rate)
    res5 = extract_features(pitch_data)
    result = np.vstack((result, res5)) 
    
    #speed up
    higher_speed_data = higher_speed(data)
    res6 = extract_features(higher_speed_data)
    result = np.vstack((result, res6))
    
    #speed down
    lower_speed_data = higher_speed(data)
    res7 = extract_features(lower_speed_data)
    result = np.vstack((result, res7))
    
    return result

def predict_emotion(file_name):
    labels = ['angry','calm','disgust','fear','happy','nuetral', 'sad', 'suprise']
    
    
    loaded_model = load_model("./trained_models/total_model.h5", compile = True)
    scaler = StandardScaler()
    encoder = OneHotEncoder()
    
    # path = main()
    file_location = f"./uploads/audio_records/{file_name}"
    print(file_location)

    res = get_features(file_location)
    scaler.fit_transform(res)
    x_res = scaler.transform(res)
    x_res = np.expand_dims(x_res, axis=2)
    
    pred_res = loaded_model.predict(x_res)
    pred_label = pred_res.argmax(axis=-1)
    
    print('pred_label',pred_label)
    
    final_result = ""
    if pred_label[6] == 0:
        final_result = "Angry"
    elif pred_label[6] == 1:
        final_result = "Calm"
    elif pred_label[6] == 2:
        final_result = "Disgust"
    elif pred_label[6] == 3:
        final_result = "Fear"
    elif pred_label[6] == 4:
        final_result = "Happy"
    elif pred_label[6] == 5:
        final_result = "Nuetral"
    elif pred_label[6] == 6:
        final_result = "Sad"
    elif pred_label[6] == 7:
        final_result = "Suprise"
    
    print('final_result',final_result)
    return final_result
