import os
import math
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime as dt, timedelta

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from . import train_model as tm
from . import save_csv as sc


# 最新の昨日のデータがあるか判定
def is_yesterday(file_path):
    df = pd.read_csv(file_path, header=1, encoding='shift-jis')
    last_date = list(df.tail(1).DATE)[0].split('/')
    dt_last = dt(int(last_date[0]), int(last_date[1]), int(last_date[2])).date()
    dt_yday = dt.today().date() - timedelta(1)
    print('[INFO: Last day in csv file is '+ str(dt_last) +'.]')
    if dt_last == dt_yday:
        print('[INFO: Yesterday csv file is existing.]')
        return True
    else:
        print('[INFO: No yesterday data in csv file.]')
        return False


# 電力推定
def predict_power():
    print('[INFO: Start predict]')
    is_file = os.path.isfile(sc.CSV_PATH)
    if not is_file or not is_yesterday(sc.CSV_PATH):
        sc.save_csv()
    trainx, trainy, testx, testy, scaler = tm.create_dataset(tm.look_back)

    hdf5 = os.listdir(tm.MODEL_DIR)[0]
    model = load_model(os.path.join(tm.MODEL_DIR, hdf5))
    testpredict = model.predict(testx)
    testpredict = scaler.inverse_transform(testpredict)

    return testpredict[-1][0]
