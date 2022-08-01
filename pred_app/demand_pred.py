import os
import math
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime as dt, timedelta

from sklearn.metrics import mean_squared_error
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
    # print(dt_last)
    dt_yday = dt.today().date() - timedelta(1)
    # print(dt_yday)
    if dt_last == dt_yday:
        return True
    else:
        return False


# 学習結果
def score(model, trainx, testx, scaler):
    # テストデータに対する予測（評価のため訓練データも）
    trainpredict = model.predict(trainx)
    testpredict = model.predict(testx)
    
    # 正規化を元に戻す
    trainpredict = scaler.inverse_transform(trainpredict)
    trainy = scaler.inverse_transform([trainy])
    testpredict = scaler.inverse_transform(testpredict)
    testy = scaler.inverse_transform([testy])
    
    # 平均二乗誤差のルートで評価
    trainscore = math.sqrt(mean_squared_error(trainy[0], trainpredict[:,0]))
    testscore = math.sqrt(mean_squared_error(testy[0], testpredict[:,0]))

    return trainpredict, trainy, trainscore, testpredict, testy, testscore


# 電力推定
def predict_power():
    is_file = os.path.isfile(sc.CSV_PATH)
    if not is_file or not is_yesterday(sc.CSV_PATH):
        sc.save_csv()
    trainx, trainy, testx, testy, scaler = tm.create_dataset(tm.look_back)

    hdf5 = os.listdir(tm.MODEL_DIR)[0]
    model = load_model(hdf5)
    testpredict = model.predict(testx)
    testpredict = scaler.inverse_transform(testpredict)

    return testpredict[-1][0]
