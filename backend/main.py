from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from model_loader import load_models
from fastapi.middleware.cors import CORSMiddleware 
from mangum import Mangum

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Atau ganti dengan domain frontend seperti 'http://localhost:3000'
    allow_credentials=True,
    allow_methods=["*"],  # Memungkinkan semua metode HTTP
    allow_headers=["*"],  # Memungkinkan semua header
)

MODEL_SMA_URL = "https://drive.google.com/uc?export=download&id=1OXKXBEeXDXKgSo0J544En0qpxzHCjGZM"
MODEL_S1_URL = "https://drive.google.com/uc?export=download&id=1buKNW7-TFYMyKC-qv3sLBsywkAhQNXpC"
MODEL_SMA_PATH = "model_sma.pkl"
MODEL_S1_PATH = "model_s1.pkl"

def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        r = requests.get(url)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"{path} downloaded.")
    else:
        print(f"{path} already exists, skipping download.")

@app.on_event("startup")
def startup_event():
    download_file(MODEL_SMA_URL, MODEL_SMA_PATH)
    download_file(MODEL_S1_URL, MODEL_S1_PATH)
    global model_sma, model_s1
    from model_loader import load_models
    model_sma, model_s1 = load_models(MODEL_SMA_PATH, MODEL_S1_PATH)

class InputData(BaseModel):
    responses: list[int]  # List berisi 0 atau 1

@app.post("/predict")
async def predict(data: InputData):
    if not data.responses or not all(r in [0, 1] for r in data.responses):
        raise HTTPException(status_code=400, detail="Invalid input format")

    try:
        input_array = np.array(data.responses).reshape(1, -1)
        pred_sma = model_sma.predict(input_array)[0]
        pred_s1 = model_s1.predict(input_array)[0]
        return {
            "sma": pred_sma,
            "s1": pred_s1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
