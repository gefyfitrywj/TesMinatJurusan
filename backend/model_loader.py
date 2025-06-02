import joblib

def load_models(path_sma, path_s1):
    model_sma = joblib.load(path_sma)
    model_s1 = joblib.load(path_s1)
    return model_sma, model_s1
