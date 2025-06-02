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

model_sma, model_s1 = load_models()

class InputData(BaseModel):
    responses: list[int]  # 0 atau 1

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
