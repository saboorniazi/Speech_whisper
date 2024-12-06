from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from fastapi.middleware.cors import CORSMiddleware
import whisper
import io
import torch
import ssl
# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context
# Initialize FastAPI app


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load Whisper model
model = whisper.load_model("small").to(device) # You can change the model to "small", "medium", etc.

# Helper function to process the audio
def transcribe_audio(audio_data: bytes) -> str:
    # Load the audio file using pydub
    audio = AudioSegment.from_file(io.BytesIO(audio_data))
    
    # Save the audio as a temporary WAV file (whisper works with WAV or MP3)
    temp_file = "/tmp/temp_audio.wav"
    audio.export(temp_file, format="wav")
    
    # Use Whisper to transcribe the audio
    result = model.transcribe(temp_file)
    print(result)
    # Return the transcribed text
    return result["text"]

# Endpoint to handle audio upload and transcription
@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Read the file data
        audio_data = await file.read()
        
        # Transcribe the audio
        transcribed_text = transcribe_audio(audio_data)
        
        return JSONResponse(content={"message": "Transcription successful", "transcription": transcribed_text})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)