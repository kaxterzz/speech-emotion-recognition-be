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
from keras.models import Sequential
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

    fig = plt.figure(figsize=(17, 5))

    fig.add_subplot(121)
    plt.title('Count of Females Emotions', size=16)
    sns.countplot(Females.labels, order = order)
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)

    fig.add_subplot(122)
    plt.title('Count of Males Emotions', size=16)
    sns.countplot(Males.labels, order = order)
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)

    plt.show()

def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title(f'Waveplot for audio with {e} emotion', size=15)
    librosa.display.waveplot(data, sr=sr)
    plt.show()

# emotion='Angry'
# path = './RAVDESS/Actor_01/03-01-05-01-01-01-01.wav'
# data, sampling_rate = librosa.load(path)
# create_waveplot(data, sampling_rate, emotion)
# Audio(path)

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


# taking any example and checking for techniques.
# path = path = './data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav'
# data, sample_rate = librosa.load(path)

# plt.figure(figsize=(10,3))
# x = noise(data)
# librosa.display.waveplot(y=x, sr=sample_rate)
# Audio(x, rate=sample_rate)

#sample_rate = 22050

def extract_features(data):
    
    result = np.array([])
    
    #mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=42) #42 mfcc so we get frames of ~60 ms
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    result = np.array(mfccs_processed)
     
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=3, offset=0.5, res_type='kaiser_fast') 
    
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

if not DATA_FRAMES:
    
    female_X, female_Y = [], []
    for path, emotion in zip(Females.path, Females.labels):
        features = get_features(path)
        #adding augmentation, get_features return a multi dimensional array (for each augmentation), so we have to use a loop to fill the df
        for elem in features: 
            female_X.append(elem)        
            female_Y.append(emotion)
    

    male_X, male_Y = [], []
    for path, emotion in zip(Males.path, Males.labels):
        features = get_features(path)
        for elem in features:
            male_X.append(elem)
            male_Y.append(emotion)
            
    print(f'Check shapes:\nFemale features: {len(female_X)}, labels: {len(female_Y)}\nMale features:   {len(male_X)}, labels: {len(male_Y)}')

def setup_dataframe(gender, features, labels):
    df = pd.DataFrame(features)
    df['labels'] = labels
    df.to_csv(f'{gender}_features.csv', index=False)
    
    print(f'{gender} dataframe')
    df.sample(frac=1).head()
    
    return df

if not DATA_FRAMES:
    Females_Features = setup_dataframe('Female', female_X, female_Y)
else:
    Females_Features = pd.read_csv(fem_path)

if not DATA_FRAMES:
    Males_Features = setup_dataframe('Male', male_X, male_Y)
else:
    Males_Features = pd.read_csv(mal_path)


female_X = Females_Features.iloc[: ,:-1].values
female_Y = Females_Features['labels'].values

male_X = Males_Features.iloc[: ,:-1].values
male_Y = Males_Features['labels'].values

# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()

female_Y = encoder.fit_transform(np.array(female_Y).reshape(-1,1)).toarray()
male_Y = encoder.fit_transform(np.array(male_Y).reshape(-1,1)).toarray()

nogender_X = np.concatenate((female_X, male_X))
nogender_Y = np.concatenate((female_Y, male_Y))

x_train, x_test, y_train, y_test = train_test_split(nogender_X, nogender_Y, random_state=0, test_size=0.20, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

x_trainF, x_testF, y_trainF, y_testF = train_test_split(female_X, female_Y, random_state=0, test_size=0.20, shuffle=True)
x_trainF.shape, y_trainF.shape, x_testF.shape, y_testF.shape

x_trainM, x_testM, y_trainM, y_testM = train_test_split(male_X, male_Y, random_state=0, test_size=0.20, shuffle=True)
x_trainM.shape, y_trainM.shape, x_testM.shape, y_testM.shape

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_trainF = scaler.fit_transform(x_trainF)
x_testF = scaler.transform(x_testF)

x_trainM = scaler.fit_transform(x_trainM)
x_testM = scaler.transform(x_testM)

x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train.shape, y_train.shape , x_test.shape , y_test.shape

x_trainF = np.expand_dims(x_trainF, axis=2)
x_testF = np.expand_dims(x_testF, axis=2)
x_trainF.shape, y_trainF.shape, x_testF.shape, y_testF.shape

x_trainM = np.expand_dims(x_trainM, axis=2)
x_testM = np.expand_dims(x_testM, axis=2)
x_trainM.shape, y_trainM.shape, x_testM.shape, y_testM.shape

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    
    def build_model(in_shape):
        
        model=Sequential()
        model.add(Conv1D(256, kernel_size=6, strides=1, padding='same', activation='relu', input_shape=(in_shape, 1)))
        model.add(AveragePooling1D(pool_size=4, strides = 2, padding = 'same'))

        model.add(Conv1D(128, kernel_size=6, strides=1, padding='same', activation='relu'))
        model.add(AveragePooling1D(pool_size=4, strides = 2, padding = 'same'))

        model.add(Conv1D(128, kernel_size=6, strides=1, padding='same', activation='relu'))
        model.add(AveragePooling1D(pool_size=4, strides = 2, padding = 'same'))
        model.add(Dropout(0.2))

        model.add(Conv1D(64, kernel_size=6, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=4, strides = 2, padding = 'same'))
        
        model.add(Flatten())
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(units=8, activation='softmax'))
        model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
          
        
        return model

def model_build_summary(mod_dim, tr_features, val_features, val_labels):
    model = build_model(mod_dim)
    model.summary()
    
    score = model.evaluate(val_features, val_labels, verbose = 1)
    accuracy = 100*score[1]
    
    return model

rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=4, min_lr=0.000001)

batch_size = 32
n_epochs = 75

def show_graphs(history):
    epochs = [i for i in range(n_epochs)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    test_acc = history.history['val_accuracy']
    test_loss = history.history['val_loss']

    fig.set_size_inches(30,12)
    ax[0].plot(epochs , train_loss , label = 'Training Loss')
    ax[0].plot(epochs , test_loss , label = 'Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
    ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    plt.show()

total_model = model_build_summary(x_train.shape[1], x_train, x_test, y_test)
female_model = model_build_summary(x_trainF.shape[1], x_trainF, x_testF, y_testF)
male_model = model_build_summary(x_trainM.shape[1], x_trainM, x_testM, y_testM)

history = total_model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_data=(x_test, y_test), callbacks=[rlrp])
total_model.save("total_model.h5")

female_history = female_model.fit(x_trainF, y_trainF, batch_size=batch_size, epochs=n_epochs, validation_data=(x_testF, y_testF), callbacks=[rlrp])
female_model.save("female_model.h5")

male_history = male_model.fit(x_trainM, y_trainM, batch_size=batch_size, epochs=n_epochs, validation_data=(x_testM, y_testM), callbacks=[rlrp])
male_model.save("male_model.h5")

# genderless
score = total_model.evaluate(x_train,y_train, verbose = 0)
print("Mixed-gender emotions training Accuracy: {0:.2%}".format(score[1]))

score = total_model.evaluate(x_test, y_test, verbose=0)
print("Mixed-gender emotions testing Accuracy: {0:.2%}".format(score[1]))

score = female_model.evaluate(x_trainF,y_trainF, verbose = 0)
print("Female emotions training Accuracy: {0:.2%}".format(score[1]))

score = female_model.evaluate(x_testF, y_testF, verbose=0)
print("Female emotions testing Accuracy: {0:.2%}".format(score[1]))

score = male_model.evaluate(x_trainM,y_trainM, verbose = 0)
print("Male emotions training Accuracy: {0:.2%}".format(score[1]))

score = male_model.evaluate(x_testM, y_testM, verbose=0)
print("Male emotions testing Accuracy: {0:.2%}".format(score[1]))

show_graphs(history)
show_graphs(female_history)
show_graphs(male_history)

# predicting on test data.
pred_test = female_model.predict(x_testF)
y_pred = encoder.inverse_transform(pred_test)
y_test_ = encoder.inverse_transform(y_testF)

# predicting on test data.
pred_test = male_model.predict(x_testM)
y_pred = encoder.inverse_transform(pred_test)
y_test_ = encoder.inverse_transform(y_testM)

