from urllib import response
from fastapi import FastAPI
from fastapi import Depends, FastAPI, HTTPException, status, File, UploadFile, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sql_app import crud, models, schemas
from sql_app.database import SessionLocal, engine
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List
from predict_from_total_model import predict_emotion
# from predict_from_tess_model import predict_emotion
from process_emotion_name_audio import process, mainp
import moviepy.editor as moviepy
import subprocess

models.Base.metadata.create_all(bind=engine)

ACCESS_TOKEN_EXPIRE_MINUTES = 3000

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

def convert_and_split(file_location,filename):
    outputfile_name = filename[:-5]
    print('outputfile_name',outputfile_name)
    output_path = outputfile_name + ".wav"
    command = ['ffmpeg', '-i', file_location, '-f', 'segment', '-segment_time', '15', 'out%09d.wav']
    subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    
def get_db():
    with Session(engine) as db:
        yield db
        
@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    user_email_exists = crud.get_user_by_email(db, email=user.email)
    if user_email_exists:
        raise HTTPException(status_code=400, detail={"type":"email","message":"Email already registered !"})
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = crud.create_access_token(
        data={"id": user.user_id, "name": user.name, "email": user.email}, expires_delta=access_token_expires
    )

    return { "access_token": access_token, "token_type": "bearer" }


@app.post("/upload-audio-records")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"./uploads/audio_records/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    predicted_emotion = predict_emotion(file.filename)
    return {"predicted_emotion": predicted_emotion,"info": f"file '{file.filename}' saved at '{file_location}'"}

@app.post("/upload-emotion-name-audio-records")
async def create_upload_file(files: List[UploadFile]):
    transcriptions = []
    for file in files:
        file_location = f"./uploads/emotion_name_audio_records/webm/{file.filename}"
        with open(file_location, "ab") as file_object:
            file_object.write(file.file.read())
        # predicted_emotion = predict_emotion(file.filename)
        convert_and_split(file_location,file.filename)
        transcription = process("out000000000.wav")
        fname = file.filename
        info = {"filename": fname, "original_emotion": fname[19:-27], "transcription": transcription}
        transcriptions.append(info)
    print(transcriptions)
    return {"data": transcriptions}