import uvicorn
import io
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from fastapi import FastAPI, Body, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


app = FastAPI()


@app.post("/")
async def handle_f(assign: UploadFile = File(...)):
    if assign is not None:
    
        EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
        language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
        signal = language_id.load_audio(f"{assign.filename}")
        #prediction =  language_id.classify_batch(signal)
        print({"имя": assign.filename})
        