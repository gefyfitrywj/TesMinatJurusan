import joblib
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_SMA_PATH = os.path.join(BASE_DIR, 'models', 'rf_sma_model.pkl')
MODEL_S1_PATH = os.path.join(BASE_DIR, 'models', 'rf_s1_model.pkl')

def load_models():
    model_sma = joblib.load(MODEL_SMA_PATH)
    model_s1 = joblib.load(MODEL_S1_PATH)
    return model_sma, model_s1
