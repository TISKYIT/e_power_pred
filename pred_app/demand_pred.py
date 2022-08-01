import os
import urllib.request

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


############################################################################
# 設計内容
#　・0時に東京電力hpへアクセスし、csvファイルの更新を行う.
#　・csvファイルの更新後、続けて学習を実施し、モデルを更新する.
#　・今回、精度によるモデルの更新の有無は考慮しない. 学習後は必ず更新を行う.
############################################################################


# gpu無効化
os.environ["cuda_visible_devices"]="-1" 

# 東京電力URL
URL = 'https://www.tepco.co.jp/forecast/html/images/juyo-result-j.csv'

# csv path
CSV_DIR = os.path.join(os.getcwd(), 'pred_app/csv/')
CSV_NAME = os.path.basename(URL)
CSV_PATH = os.path.join(CSV_DIR, CSV_NAME)

# model path 
MODEL_DIR = os.path.join(os.getcwd(), 'pred_app/model')


# 今日のデータの存在確認
def is_today(file_path):
    df = pd.read_csv(file_path, header=1, encoding='shift-jis')
    last_date = list(df.tail(1).DATE)[0].split('/')
    dt_last = dt(int(last_date[0]), int(last_date[1]), int(last_date[2])).date()
    print(dt_last)
    dt_tday = dt.today().date()
    print(dt_tday)
    if dt_last == dt_tday:
        return True
    else:
        return False


# csv保存 ※cronで実行
def cron_save_csv():
    urllib.request.urlretrieve(URL, CSV_PATH)


# csv読込み
def read_csv(file_path):
    df = pd.read_csv(file_path, usecols=[7], header=1, encoding='shift-jis')
    dataset = df.values.astype('float32')

    return dataset


look_back=3
batch_size = 1
epochs = 100
ratio = 0.67


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
    dataset = read_csv(CSV_PATH)
    dataset, scaler = normalize(dataset)
    train, test = split_dataset(dataset, ratio=0.67)
    trainx, trainy = look_back_dataset(train, look_back)
    testx, testy = look_back_dataset(test, look_back)

    # [samples, time steps, features]へ変形
    trainx = np.reshape(trainx, (trainx.shape[0], trainx.shape[1], 1))
    testx = np.reshape(testx, (testx.shape[0], testx.shape[1], 1))

    return trainx, trainy, testx, testy, scaler


# 学習 ※cronで実行
def cron_train():
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


# モデル取得
def model_file():
    files = os.listdir(MODEL_DIR)
    last_date = max([int(file[6:14]) for file in files])
    last_files = [file for file in files if str(last_date) in file]
    best_accuracy = min([file[18:24] for file in last_files])
    best_file = glob(os.path.join(MODEL_DIR, 'epower'+str(last_date)+'_*_'+str(best_accuracy)+'.hdf5'))[0]

    return best_file


# 電力推定
def predict_power():
    is_file = os.path.isfile(CSV_PATH)
    if not is_file or not is_today(CSV_PATH):
        cron_save_csv()
    
    trainx, trainy, testx, testy, scaler = create_dataset(look_back)
    best_file = model_file()
    model = load_model(best_file)
    testpredict = model.predict(testx)
    testpredict = scaler.inverse_transform(testpredict)

    return testpredict[-1][0]
