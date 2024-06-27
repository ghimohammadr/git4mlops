import pathlib
import os
import train, ML_package


PACKAGE_ROOT = pathlib.Path(ML_package.__file__).resolve().parent
TRAIN_ROOT = pathlib.Path(train.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

DATA_FILE = 'speed_volume_test.xlsx'

MODEL_NAME = 'trafficprediction.pkl'
SAVE_MODEL_PATH = os.path.join(TRAIN_ROOT, 'trained_model')

TARGET = 'Flow'

FEATURES = ['Flow', 'speed']

DROP_FEATURES = ['Date']

NROM_var = 1
TIME_HORIZON_parm = 1
DELAY_parm = 8
BATCH_SIZE = 2880

FEATURES_TO_ENCODE = ['Flow', 'speed']