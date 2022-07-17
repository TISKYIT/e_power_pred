from re import I
import requests
import os

import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM


# 東京電力のCSVダウンロード元URL
URL = 'http://www.tepco.co.jp/forecast/html/images/juyo-result-j.csv'

# csvファイルの保存先
FILE_DIR = os.path.join(os.getcwd(), 'pred_app/csv/')

# print("現在のカレントパス:{0}".format(os.getcwd()))
# print("FILE_DIRのパス:{0}".format(FILE_DIR))


def get_csv():
    """ 東京電力URLからcsvファイルをダウンロードしcsvディレクトリへ保存する
        
        TODO: ファイルの中身を確認して本日の情報があれば取得しに行かない様に修正した方が良い
    
    :raises exc: 東京電力URLへファイル取得した時のレスポンスエラー
    """
    response = requests.get(URL)
    try:
        response_status = response.raise_for_status()
    except Exception as exc:
        print('Error:{}'.format(exc))
    if response_status == None:
        file = open(os.path.join(FILE_DIR, os.path.basename(URL)), 'wb')

        for chunk in response.iter_content(100000):
            file.write(chunk)
        file.close()
        print('file saved')


def get_dataset(file_path):
    """ 引数に保存されたcsvをdataframeへ変換

        下記の様なカラムで構成されているため、8番目の使用率ピーク時需要電力のみ取得
        DATE,TIME,曜日,実績(万kW),ピーク時供給力,使用率,使用率ピーク時時間帯,使用率ピーク時需要電力,使用率ピーク時供給力,使用率ピーク時使用率

        補足: データがshift-jisでしたがそのままにしてあります。
    
    :param file_path: csvファイルへのパス
    :return 使用率ピーク時需要電力のみのデータセット
    :rtype: pandas.dataframe
    """
    dataframe = pd.read_csv(file_path, usecols=[7], header=1, encoding='shift-jis')
    dataset = dataframe.values.astype('float32')

    return dataset


def normalize(dataset):
    """ 引数で与えられた時系列データを0～1の値へ正規化
    
    :param dataset: 需要電力のみのデータが入ったデータセット
    :return: 0~1の値に正規化されたデータセット
    :rtype: pandas.dataframe
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    return dataset, scaler


def split_dataset(dataset, ratio=0.67):
    """ datasetを訓練データと検証データに分類する

    :param dataset: 0～1に正規化された時系列データ
    :param ratio: 訓練データの割合
    :return train: 訓練データ
    :return test: 検証データ
    :rtype train: pandas.dataframe
    :rtype test: pandas.dataframe
    """
    train_size = int(len(dataset) * ratio)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    return train, test


def look_back_dataset(dataset, look_back=1):
    """ X=[data[t-look_back],...,data[t-1]], Y=data[t]となるデータセットに変換
    
    :param dataset: 0～1に正規化された時系列データ
    :param look_back: いくつの過去のデータとセットにするかを決める変数
    :return np.array(dataX): [data[t-look_back],...,data[t-1]]の形に変換された入力データセット
    :return np.array(dataY): data[t]の形に変換された出力データセット
    :rtype: np.array(dataX): np.array
    :rtype: np.array(dataY): np.array
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    
    return np.array(dataX), np.array(dataY)


def create_dataset(look_back=1):
    """ 訓練データセットとテストデータセットを作成
    :param train: 0～1に正規化された時系列データ
    :param test: 0～1に正規化された時系列データ
    :return trainX: 
    :return trainY: 
    :return testX:
    :return testY:
    :rtype: np.array(dataX): np.array
    """
    file_path = os.path.join(FILE_DIR, os.path.basename(URL))
    dataset = get_dataset(file_path)
    dataset, scaler = normalize(dataset)
    train, test = split_dataset(dataset, ratio=0.67)
    trainX, trainY = look_back_dataset(train, look_back)
    testX, testY = look_back_dataset(test, look_back)

    # [samples, time steps, features]へ変形
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    return trainX, trainY, testX, testY, scaler


def set_model(look_back=1):
    # LSTMにDenseを接続し、数値を予測（MSEで評価）
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1))) # input_shape=(系列長T, x_tの次元), output_shape=(units,)
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def train(model, trainX, trainY):
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    return model


def get_score(model, trainX, trainY, testX, testY, scaler): 
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
    print('Train RMSE: %.2f' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test RMSE: %.2f' % (testScore))

    return trainPredict, trainY, trainScore, testPredict, testY, testScore


def get_pred():
    # 学習/検証データセット作成
    trainX, trainY, testX, testY, scaler = create_dataset(look_back=3)
    # モデル作成
    model = set_model(look_back=3)
    # 学習
    model = train(model, trainX, trainY)
    # スコア
    trainPredict, trainY, trainScore, testPredict, testY, testScore = get_score(model, trainX, trainY, testX, testY, scaler)

    return testPredict[-1][0]