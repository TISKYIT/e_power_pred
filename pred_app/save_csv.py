import os
import urllib.request


# 東京電力URL
URL = 'https://www.tepco.co.jp/forecast/html/images/juyo-result-j.csv'

# csv path
CSV_DIR = os.path.join(os.getcwd(), 'pred_app/csv/')
CSV_NAME = os.path.basename(URL)
CSV_PATH = os.path.join(CSV_DIR, CSV_NAME)


# csv保存 ※JST:00:00:00にHeroku Schedulerで実行
# TODO: Herokuが東電からアクセス拒否されている可能性あり.
# TODO: 東電のcsvが更新される日時を調べる必要がある.頻繁にsave_csvが走る.
def save_csv():
    try:
        urllib.request.urlretrieve(URL, CSV_PATH)
        print('[INFO: Csv file is uploaded.]')
    except:
        print('[WARNING: Csv file could not saved.]')
        pass


if __name__ == '__main__':
    save_csv()