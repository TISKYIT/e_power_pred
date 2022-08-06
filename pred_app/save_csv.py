import os
import urllib.request


# 東京電力URL
URL = 'https://www.tepco.co.jp/forecast/html/images/juyo-result-j.csv'

# csv path
CSV_DIR = os.path.join(os.getcwd(), 'pred_app/csv/')
CSV_NAME = os.path.basename(URL)
CSV_PATH = os.path.join(CSV_DIR, CSV_NAME)


# csv保存 ※JST:00:00:00にHeroku Schedulerで実行
def save_csv():
    try:
        urllib.request.urlretrieve(URL, CSV_PATH)
    except:
        pass


if __name__ == '__main__':
    save_csv()