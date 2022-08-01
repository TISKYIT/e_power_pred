import os
import math
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from . import save_csv as sc


# model path 
MODEL_DIR = os.path.join(os.getcwd(), 'pred_app/model')


ratio = 0.67
look_back=3
batch_size = 1
epochs = 100


# csv読込み
def read_csv(file_path):
    df = pd.read_csv(file_path, usecols=[7], header=1, encoding='shift-jis')
    dataset = df.values.astype('float32')

    return dataset


# 正規化実行
def normalize(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    return dataset, scaler


# データ分割（学習/検証）
def split_dataset(dataset, ratio=0.67):
    train_size = int(len(dataset) * ratio)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    return train, test


# rnn用データセットへ変換
def look_back_dataset(dataset, look_back=1):
    datax, datay = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        datax.append(a)
        datay.append(dataset[i + look_back, 0])
    
    return np.array(datax), np.array(datay)


# 学習/検証データセット作成
def create_dataset(look_back):
    dataset = read_csv(sc.CSV_PATH)
    dataset, scaler = normalize(dataset)
    train, test = split_dataset(dataset, ratio=0.67)
    trainx, trainy = look_back_dataset(train, look_back)
    testx, testy = look_back_dataset(test, look_back)

    # [samples, time steps, features]へ変形
    trainx = np.reshape(trainx, (trainx.shape[0], trainx.shape[1], 1))
    testx = np.reshape(testx, (testx.shape[0], testx.shape[1], 1))

    return trainx, trainy, testx, testy, scaler


# 学習 ※JST:00:00:05にHeroku Schedulerで実行
def train():
    # 学習/検証データ作成
    trainx, trainy, testx, testy, scaler = create_dataset(look_back)

    # lstmにdenseを接続し、数値を予測（mseで評価）
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1))) # input_shape=(系列長t, x_tの次元), output_shape=(units,)
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    str_today = dt.today().strftime('%Y%m%d')

    chkpt = os.path.join(MODEL_DIR, 'epower'+str_today+'_{epoch:02d}_{val_loss:.4f}.hdf5')
    cp_cb = ModelCheckpoint(filepath=chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit(x=trainx, y=trainy, validation_data=(testx, testy), batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[cp_cb, es_cb])

    # ここから関数化
    today_models = glob(os.path.join(MODEL_DIR, '*'+str_today+'*'))
    max_epoch = max([int(file[-14:-12]) for file in today_models])
    ep = '{ep:02}'.format(ep=max_epoch)

    # 今日の最良モデルのみ残す
    best_model = glob(os.path.join(MODEL_DIR, '*_'+ep+'_*'))
    remove_models = [os.remove(file) for file in today_models if not '_'+ep+'_' in file]

    # 今日の最良モデルと、過去の最良モデルを比較して精度の良い方を残す
    models = glob(os.path.join(MODEL_DIR, '*'))
    results = np.array([score(load_model(model), trainx, trainy, testx, testy, scaler) for model in models], dtype=object)
    remove_which = np.argmax(results[:, 5:6])
    os.remove(models[remove_which])


# 学習結果
def score(model, trainx, trainy, testx, testy, scaler):
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


if __name__ == '__main__':
   train()