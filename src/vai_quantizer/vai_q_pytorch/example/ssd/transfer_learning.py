from ssd_model import SSD
from ssd_model import MultiBoxLoss
from train import train_model

from ssd_model import make_datapath_list, VOCDataset, DataTransform, Anno_txt2list, od_collate_fn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


## DataLoaderの作成
# ファイルパスlistを取得　#TODO

# Datasetの作成
coco_classes = ['person','bicycle', 'car']
color_mean=(104, 117, 123) #TODO
input_size = 300

train_img_list, train_anno_list, val_img_list, val_anno_list=make_datapath_list(train_img_path="/home/ubuntu/Chipathon/train/ssd/data/COCO_data/train.txt", 
                                                                                val_img_path="/home/ubuntu/Chipathon/train/ssd/data/COCO_data/val.txt")
# class名　TODO
train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_txt2list(coco_classes))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_txt2list(coco_classes))

# DataLoaderを作成する
batch_size = 32

train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}



## modelの設定

ssd_cfg = {
    'num_classes': 21,  # 背景クラスを含めた合計クラス数
    'input_size': 300,  # 画像の入力サイズ
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

net = SSD(phase="train", cfg=ssd_cfg)
net_weights = torch.load('./weight/ssd300_50.pth',
                         map_location={'cuda:0': 'cpu'})
net.load_state_dict(net_weights)

# GPUが使えるかを確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("使用デバイス：", device)
print('ネットワーク設定完了：学習済みの重みをロードしました')

# 層の付け替え
net.conf[0]=nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
net.conf[1]=nn.Conv2d(1024, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
net.conf[2]=nn.Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
net.conf[3]=nn.Conv2d(256, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
net.conf[4]=nn.Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
net.conf[5]=nn.Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

layers=[]
for name, param in net.named_parameters():
  layers.append(name)

update_param_names=layers[-12:]
# 転移学習で学習させるパラメータを、変数params_to_updateに格納する
params_to_update = []

# 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
        print(name)
    else:
        param.requires_grad = False




## 損失関数の設定
criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)

## 最適化手法の設定
optimizer = optim.SGD(net.parameters(), lr=1e-3,
                      momentum=0.9, weight_decay=5e-4)

## 学習
num_epochs=10
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, 
            save_result_path="/home/ubuntu/Chipathon/train/ssd/result/log_output.csv", 
            save_weight_path="/home/ubuntu/Chipathon/train/ssd/weight/trained/")