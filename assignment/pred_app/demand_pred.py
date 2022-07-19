import os
from re import I
import urllib3

import math
import numpy as np
import pandas as pd
from datetime import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

# 設計
# cronで3時間ずつCSVを確認する.
# CSVが更新されたタイミングで、学習を実行する



# 東京電力のCSVダウンロード元URL
URL = 'http://www.tepco.co.jp/forecast/html/images/juyo-result-j.csv'

# csv
CSV_DIR = os.path.join(os.getcwd(), 'pred_app/csv/')
MODEL_DIR = os.path.join(os.getcwd(), 'pred_app/model')
CSV_NAME = os.path.basename(URL)
CSV_PATH = os.path.join(CSV_DIR, CSV_NAME)


def is_today(file_path):
    df = pd.read_csv(file_path, header=1, encoding='shift-jis')
    last_date_str = list(df.tail(1).DATE)[0]
    dt_last = dt.strptime(last_date_str, '%Y/%m/%d').date()
    dt_tday = dt.date.today()
    if dt_last == dt_tday:
        return True
    else:
        return False


def save_csv(url, file_name):
    urllib3.request.urlretrieve(url, file_name)


#def predict_power():
#    is_file = os.path.isfile(CSV_PATH)
#    if not is_file or not is_today(CSV_PATH):
#        save_csv(URL, CSV_NAME)
#    y = model.predict(x)
#    y = scaler.inverse_transform(y)
#
#    return y


def read_csv(file_path):
    df = pd.read_csv(file_path, usecols=[7], header=1, encoding='shift-jis')
    dataset = df.values.astype('float32')

    return dataset


def normalize(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    return dataset, scaler


def split_dataset(dataset, ratio=0.67):
    train_size = int(len(dataset) * ratio)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    return train, test


def look_back_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    
    return np.array(dataX), np.array(dataY)


def create_dataset(look_back=1):
    dataset = read_csv(CSV_PATH)
    dataset, scaler = normalize(dataset)
    train, test = split_dataset(dataset, ratio=0.67)
    trainX, trainY = look_back_dataset(train, look_back)
    testX, testY = look_back_dataset(test, look_back)

    # [samples, time steps, features]へ変形
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    return trainX, trainY, testX, testY, scaler


def train(trainX, trainY, look_back=1):
    batch_size = 1
    epochs = 10

    # LSTMにDenseを接続し、数値を予測（MSEで評価）
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1))) # input_shape=(系列長T, x_tの次元), output_shape=(units,)
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    chkpt = os.path.join(MODEL_DIR, 'esc50_.{epoch:02d}_{loss:.4f}.hdf5')
    cp_cb = ModelCheckpoint(filepath = chkpt, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    es_cb = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[cp_cb, es_cb])

    print('学習されない')
    # model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)

def score(model, trainX, trainY, testX, testY, scaler):
    # テストデータに対する予測（評価のため訓練データも）
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    # 正規化を元に戻す
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
    # 平均二乗誤差のルートで評価
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

    return trainPredict, trainY, trainScore, testPredict, testY, testScore