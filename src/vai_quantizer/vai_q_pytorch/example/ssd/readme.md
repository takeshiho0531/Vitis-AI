# SSDの転移学習
すでにVOC2012で学習済みのSSDの重みと、与えられたperson/bike/carのアノテーションがされた与えられたデータを用いて転移学習を行うためのscriptです。

## 環境
学習にはgpuを使っています。

AWSのEC2インスタンスを利用していて、使ったAMIは
- AMI ID: ami-0d60b9becafb5eac6
- AMI名: Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04) 20230627

インスタンスにsshで通信したあと
```
source activate pytorch
```
でpytorchという名のconda環境をactivateさせ、このレポジトリをcloneします。
```
cd /home/ubuntu

git clone https://github.com/takeshiho0531/Chipathon.git
```

poetry環境をactivateさせます。
```
cd Chipason/train/ssd

poetry install
```
## 準備
[学習済みのssdの重み ssd300_50.pth](https://drive.google.com/open?id=1_zTTYQ2j0r-Qe3VBbHzvURD0c1P2ZSE9)を置きます。

転移学習後の重みの格納先も作ります。(epochごとに重みが生成される)
```
cd Chipathon/train/ssd

mkdir weight  # ここに学習済み重みを置く

mkdir trained # ここに転移学習された重みが格納される
```


データを置きます
```
cd Chipathon/train/ssd

mkdir data
```
`data`のディレクトリに[COCO_data](https://drive.google.com/drive/folders/1yhyr7TcGmXeiRtVHytvCeBaZyJMkTUhK?usp=drive_link) を置き、`COCO_data`の直下に[train.txt](https://drive.google.com/file/d/1xHwzu9EkDXjf0lbnj4ejCeT8WBwbfLfy/view?usp=drive_link)と[val.txt](https://drive.google.com/file/d/1DUbGEyeKzfXgYWlRbzxOg3zOezqwWPWa/view?usp=drive_link)も置きます。

その後、`data_organizing`ディレクトリの中にあるscriptを実行します。詳細は[data_organizing/readme.md](https://github.com/takeshiho0531/Chipathon/tree/train/train/ssd/data_organizing#readme)


## 実行方法
```
cd Chipathon/train/ssd

poetry run python3 transfer_learning.py
```

## 工夫/注意点
- 転移学習させるときのclassはperson/bike/carの3つですが、背景を合わせてnum_classes=4です。
- 学習済み重みをsetする際にはnum_classesが元のまま21になるようになってますが、転移学習にlayerを付け替えると[同時にnum_classesが4に変わるように](https://github.com/takeshiho0531/Chipathon/blob/train/train/ssd/ssd_model.py#L748)してます。

## 参考
[つくりながら学ぶ! PyTorchによる発展ディープラーニング](https://www.amazon.co.jp/dp/4839970254/)と[そのgithub](https://github.com/YutaroOgawa/pytorch_advanced/tree/master)



