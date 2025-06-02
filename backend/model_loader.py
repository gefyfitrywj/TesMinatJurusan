import pickle

def load_models(model_sma_path, model_s1_path):
    with open(model_sma_path, "rb") as f:
        model_sma = pickle.load(f)
    with open(model_s1_path, "rb") as f:
        model_s1 = pickle.load(f)
    return model_sma, model_s1
