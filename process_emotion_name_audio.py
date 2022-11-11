import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import speech_recognition as sr

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def process(filename):
    data = wavfile.read(filename)
    framerate = data[0]
    sounddata = data[1]
    time = np.arange(0,len(sounddata))/framerate
    input_audio, _ = librosa.load(filename, sr=16000)
    input_values = tokenizer(input_audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    print('transcription',transcription)
    return transcription
    

    
def mainp(filename):
    r = sr.Recognizer()
    full_path = "./uploads/emotion_name_audio_records/"+filename
    with sr.AudioFile(full_path) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)
        print(text)