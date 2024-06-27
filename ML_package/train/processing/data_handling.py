import os
import pandas as pd
import joblib
from train.config import config

# LOad the dataset
def load_dataset(data_name):
    filepath = os.path.join(config.DATAPATH, data_name)
    
    df = pd.read_excel(filepath)
    data = pd.DataFrame(data=df)

    return data


# Serialization
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME}")

# Deserialization
def load_pipeline(pipeline_to_load):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print("Model has been loaded")

    return model_loaded
