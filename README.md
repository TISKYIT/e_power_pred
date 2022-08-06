# e_power_pred

## 実装内容
 - 東京電力ホームページ内の最大電力実績カレンダーから、毎日の使用率ピーク時需要電力を取得する.
   https://www.tepco.co.jp/forecast/html/images/juyo-result-j.csv

 - 過去3日分の使用率ピーク時需要電力から今日の使用率ピーク時需要電力を推定する.(RNNによる推定)

 - 毎日午前0時に上記需要電力のcsvファイルを取得する.

 - 毎日午前0時30分に学習を実行する.

 - <p style="margin-bottom: 2em">学習後に得られた最も精度の高いモデルと、前日の最も精度の高いモデルを比較し、精度の高いモデルを保存する. </p>

## Herokuへデプロイ
 - WSLのインストール方法

 - HerokuCLIのインストール方法

 - GitHubアカウントの作成

 - HerokuアプリとGitからcloneしたアプリの紐づけ

 - Herokuアプリのデプロイ

 - ログの確認

 - schedulerの設定方法