# Speech Emotion Recognition for Children with Autism Spectrum Disorder

This project implements a speech emotion recognition system using deep learning and classical machine learning techniques. It is designed to help identify human emotions from speech, with a focus on supporting children with Autism Spectrum Disorder.

## Features

- Audio data preprocessing and augmentation
- Feature extraction using `librosa`
- Model training with Keras (CNN-based models)
- Emotion prediction API using FastAPI
- User management and authentication
- Speech-to-text transcription using Wav2Vec2
- Data visualization and analysis

## Project Structure

```
.
├── main.py                        # FastAPI backend and API endpoints
├── train_model.py                 # Model training scripts
├── main_data_train.py             # Data preprocessing and feature extraction
├── data_train_2.py                # Additional data processing
├── predict_from_tess_model.py     # Prediction using TESS-trained model
├── predict_from_total_model.py    # Prediction using combined model
├── process_emotion_name_audio.py  # Speech-to-text transcription
├── requirements.txt               # Python dependencies
├── sql_app/                       # Database models and CRUD logic
├── trained_models/                # Saved Keras models
├── docs/                          # Documentation and wireframes
└── ipynb_files/                   # Jupyter notebooks (if any)
```

## Setup

1. **Clone the repository**

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Prepare the datasets**
   - Place the required datasets (TESS, RAVDESS, SAVEE, CREMA-D) in the appropriate folders as referenced in the code.

4. **Train the models**
   - Run the training script to generate models:
     ```sh
     python train_model.py
     ```

5. **Start the API server**
   ```sh
   uvicorn main:app --reload
   ```

6. **API Usage**
   - Upload audio files to `/upload-audio-records` to get emotion predictions.
   - Upload audio for transcription to `/upload-emotion-name-audio-records`.

## Requirements

See [requirements.txt](requirements.txt) for the full list.

## Notable Files

- [main.py](main.py): FastAPI app and endpoints
- [train_model.py](train_model.py): Model training pipeline
- [main_data_train.py](main_data_train.py): Data loading and feature extraction
- [predict_from_total_model.py](predict_from_total_model.py): Main prediction logic
- [process_emotion_name_audio.py](process_emotion_name_audio.py): Speech-to-text transcription

## License

This project is for academic and research purposes.

---

**References:**
- [librosa](https://librosa.org/)
- [Keras](https://keras.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
