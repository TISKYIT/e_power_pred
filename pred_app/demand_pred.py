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


# 最新データがあるか判定
def is_latest(file_path):
    df = pd.read_csv(file_path, header=1, encoding='shift-jis')
    last_str = list(df.tail(1).DATE)[0]
    # CSVデータの最新日
    last_dt = dt.strptime(last_str, '%Y/%m/%d').date()
    tody_dt = dt.today().date()
    print('[INFO: Last day in csv file is '+ str(last_dt) +'.]')
    if tody_dt - last_dt == timedelta(days=2):
        # 一昨日が最新
        print('[INFO: Latest day is day after yesterday.]')
        return 2
    elif tody_dt - last_dt == timedelta(days=1): 
        # 昨日が最新
        print('[INFO: Latest day is yesterday.]')
        return 1
    else:
        print('[INFO: No latest data in csv file.]')
        return False


# 電力推定
def predict_power():
    print('[INFO: Start predict]')
    is_file = os.path.isfile(sc.CSV_PATH)
    # ファイルが無い,古い場合はcsvを再取得
    if not is_file:
        sc.save_csv()
    if not is_latest(sc.CSV_PATH):
        sc.save_csv()

    hdf5 = os.listdir(tm.MODEL_DIR)[0]
    model = load_model(os.path.join(tm.MODEL_DIR, hdf5))
    
    trainx, trainy, testx, testy, scaler = tm.create_dataset(tm.look_back)

    testpredict = model.predict(testx)
    testpredict = scaler.inverse_transform(testpredict)
    # 一昨日のデータから今日を予測
    if is_latest(sc.CSV_PATH) == 2:
        # テストデータの末尾から入力データを作成
        testx_2d = testx[-1, 1:, :]
        testy_2d = testy[-1].reshape(1,1)
        predx = np.block([[[testx_2d], [testy_2d]]])

        # 入力チェック
        # print("testx_2d(testx from 3d to 2d): \n{0}\n{1}".format(testx_2d, testx_2d.shape))
        # print("testy_2d(testy from 1d to 2d): \n{0}\n{1}".format(testy_2d, testy_2d.shape))
        # print("predx: \n{0}".format(predx))

        # 一昨日から昨日を予測したデータから入力データを作成 
        predx_2d = predx[-1, 1:, :]
        pred = model.predict(predx)
        predx_next = np.block([[[predx_2d],[pred]]])
        
        # 入力チェック
        # print("pred_2d(predx from 3d to 2d): \n{0}\n{1}".format(predx_2d, predx_2d.shape))
        # print("pred(pred is already 2d): \n{0}\n{1}".format(pred, pred.shape))
        # print("predx_next: \n{0}".format(predx_next))

        # 昨日から今日を予測
        pred_next = model.predict(predx_next) 
        pred_next = scaler.inverse_transform(pred_next)

        # 出力チェック
        # print("predy_next: \n{0}".format(pred_next))

        return pred_next[0][0]

    # 昨日のデータから今日を予測
    if is_latest(sc.CSV_PATH) == 1:
        # テストデータの末尾から入力データを作成
        testx_2d = testx[-1, 1:, :]
        testy_2d = testy[-1].reshape(1,1)
        predx = np.block([[[testx_2d], [testy_2d]]])

        # 昨日から今日を予測
        pred = model.predict(predx)
        pred = scaler.inverse_transform(pred)

        return pred[0][0] 
