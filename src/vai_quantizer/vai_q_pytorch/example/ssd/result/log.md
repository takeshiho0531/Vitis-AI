train log 
```
使用デバイス： cuda:0
ネットワーク設定完了：学習済みの重みをロードしました
conf.0.weight
conf.0.bias
conf.1.weight
conf.1.bias
conf.2.weight
conf.2.bias
conf.3.weight
conf.3.bias
conf.4.weight
conf.4.bias
conf.5.weight
conf.5.bias
使用デバイス： cuda:0
-------------
Epoch 1/1
-------------
（train）
イテレーション 10 || Loss: 7.4292 || 10iter: 16.9440 sec.
イテレーション 20 || Loss: 7.6820 || 10iter: 11.5865 sec.
イテレーション 30 || Loss: 6.3324 || 10iter: 11.8477 sec.
イテレーション 40 || Loss: 6.8519 || 10iter: 11.7546 sec.
イテレーション 50 || Loss: 5.7130 || 10iter: 11.4723 sec.
イテレーション 60 || Loss: 6.2022 || 10iter: 11.1823 sec.
イテレーション 70 || Loss: 5.6384 || 10iter: 12.0536 sec.
イテレーション 80 || Loss: 6.2326 || 10iter: 11.3672 sec.
-------------
epoch 1 || Epoch_TRAIN_Loss:565.6797 ||Epoch_VAL_Loss:0.0000
timer:  101.0190 sec.
-------------
Epoch 2/1
-------------
（train）
イテレーション 90 || Loss: 6.2585 || 10iter: 10.6535 sec.
イテレーション 100 || Loss: 6.7104 || 10iter: 11.5546 sec.
イテレーション 110 || Loss: 5.5694 || 10iter: 11.4405 sec.
イテレーション 120 || Loss: 6.1981 || 10iter: 11.2206 sec.
イテレーション 130 || Loss: 5.9180 || 10iter: 10.5149 sec.
イテレーション 140 || Loss: 6.9069 || 10iter: 11.6007 sec.
イテレーション 150 || Loss: 5.5802 || 10iter: 10.9919 sec.
イテレーション 160 || Loss: 6.1986 || 10iter: 11.8410 sec.
-------------
epoch 2 || Epoch_TRAIN_Loss:496.9719 ||Epoch_VAL_Loss:0.0000
timer:  91.6473 sec.
(pytorch) ubuntu@ip-172-31-2-109:~/Chipathon/train/ssd$ poetry run python3 transfer_learning.py
使用デバイス： cuda:0
ネットワーク設定完了：学習済みの重みをロードしました
conf.0.weight
conf.0.bias
conf.1.weight
conf.1.bias
conf.2.weight
conf.2.bias
conf.3.weight
conf.3.bias
conf.4.weight
conf.4.bias
conf.5.weight
conf.5.bias
使用デバイス： cuda:0
-------------
Epoch 1/10
-------------
（train）
イテレーション 10 || Loss: 8.2559 || 10iter: 15.5515 sec.
イテレーション 20 || Loss: 6.8092 || 10iter: 11.0068 sec.
イテレーション 30 || Loss: 6.9026 || 10iter: 11.0837 sec.
イテレーション 40 || Loss: 6.6484 || 10iter: 11.1491 sec.
イテレーション 50 || Loss: 6.2823 || 10iter: 10.8118 sec.
イテレーション 60 || Loss: 5.8155 || 10iter: 10.8965 sec.
イテレーション 70 || Loss: 5.8176 || 10iter: 10.5391 sec.
イテレーション 80 || Loss: 6.1451 || 10iter: 11.0795 sec.
-------------
epoch 1 || Epoch_TRAIN_Loss:572.1822 ||Epoch_VAL_Loss:0.0000
timer:  94.9435 sec.
-------------
Epoch 2/10
-------------
（train）
イテレーション 90 || Loss: 5.8812 || 10iter: 9.5278 sec.
イテレーション 100 || Loss: 6.1130 || 10iter: 10.8464 sec.
イテレーション 110 || Loss: 5.7873 || 10iter: 10.9642 sec.
イテレーション 120 || Loss: 6.2054 || 10iter: 10.5218 sec.
イテレーション 130 || Loss: 7.1641 || 10iter: 11.6603 sec.
イテレーション 140 || Loss: 5.8827 || 10iter: 10.6546 sec.
イテレーション 150 || Loss: 6.9705 || 10iter: 11.2139 sec.
イテレーション 160 || Loss: 5.8967 || 10iter: 11.4565 sec.
-------------
epoch 2 || Epoch_TRAIN_Loss:495.8303 ||Epoch_VAL_Loss:0.0000
timer:  88.4693 sec.
-------------
Epoch 3/10
-------------
（train）
イテレーション 170 || Loss: 5.9451 || 10iter: 9.1331 sec.
イテレーション 180 || Loss: 6.4939 || 10iter: 11.2750 sec.
イテレーション 190 || Loss: 6.0880 || 10iter: 11.6638 sec.
イテレーション 200 || Loss: 5.6601 || 10iter: 10.3928 sec.
イテレーション 210 || Loss: 6.1225 || 10iter: 10.6479 sec.
イテレーション 220 || Loss: 6.3264 || 10iter: 10.8783 sec.
イテレーション 230 || Loss: 5.9541 || 10iter: 11.4641 sec.
イテレーション 240 || Loss: 5.7890 || 10iter: 10.7229 sec.
-------------
epoch 3 || Epoch_TRAIN_Loss:489.8207 ||Epoch_VAL_Loss:0.0000
timer:  88.6054 sec.
-------------
Epoch 4/10
-------------
（train）
イテレーション 250 || Loss: 6.1348 || 10iter: 7.4228 sec.
イテレーション 260 || Loss: 5.9375 || 10iter: 10.5825 sec.
イテレーション 270 || Loss: 5.7340 || 10iter: 11.0918 sec.
イテレーション 280 || Loss: 5.8438 || 10iter: 10.4078 sec.
イテレーション 290 || Loss: 6.1090 || 10iter: 10.5177 sec.
イテレーション 300 || Loss: 5.7520 || 10iter: 10.8550 sec.
イテレーション 310 || Loss: 6.2456 || 10iter: 10.4520 sec.
イテレーション 320 || Loss: 6.0938 || 10iter: 11.2266 sec.
-------------
epoch 4 || Epoch_TRAIN_Loss:481.9476 ||Epoch_VAL_Loss:0.0000
timer:  86.1136 sec.
-------------
Epoch 5/10
-------------
（train）
イテレーション 330 || Loss: 5.8864 || 10iter: 6.3052 sec.
イテレーション 340 || Loss: 6.8787 || 10iter: 10.8074 sec.
イテレーション 350 || Loss: 5.7649 || 10iter: 11.4267 sec.
イテレーション 360 || Loss: 5.5080 || 10iter: 11.3202 sec.
イテレーション 370 || Loss: 5.8224 || 10iter: 10.7389 sec.
イテレーション 380 || Loss: 6.5739 || 10iter: 10.4349 sec.
イテレーション 390 || Loss: 5.9617 || 10iter: 10.6677 sec.
イテレーション 400 || Loss: 6.5654 || 10iter: 11.0875 sec.
-------------
epoch 5 || Epoch_TRAIN_Loss:479.9580 ||Epoch_VAL_Loss:0.0000
timer:  87.2461 sec.
-------------
Epoch 6/10
-------------
（train）
イテレーション 410 || Loss: 6.5722 || 10iter: 5.5006 sec.
イテレーション 420 || Loss: 6.0248 || 10iter: 10.2985 sec.
イテレーション 430 || Loss: 6.4139 || 10iter: 10.2746 sec.
イテレーション 440 || Loss: 6.9769 || 10iter: 9.9038 sec.
イテレーション 450 || Loss: 5.6841 || 10iter: 10.5351 sec.
イテレーション 460 || Loss: 6.1101 || 10iter: 11.5405 sec.
イテレーション 470 || Loss: 6.1843 || 10iter: 10.7216 sec.
イテレーション 480 || Loss: 5.2019 || 10iter: 11.0645 sec.
-------------
epoch 6 || Epoch_TRAIN_Loss:481.9218 ||Epoch_VAL_Loss:0.0000
timer:  85.3316 sec.
-------------
Epoch 7/10
-------------
（train）
イテレーション 490 || Loss: 4.9146 || 10iter: 3.8791 sec.
イテレーション 500 || Loss: 6.0364 || 10iter: 10.5535 sec.
イテレーション 510 || Loss: 6.2963 || 10iter: 10.4328 sec.
イテレーション 520 || Loss: 5.7767 || 10iter: 10.6317 sec.
イテレーション 530 || Loss: 6.4555 || 10iter: 12.0593 sec.
イテレーション 540 || Loss: 6.2307 || 10iter: 10.9933 sec.
イテレーション 550 || Loss: 5.7399 || 10iter: 10.2346 sec.
イテレーション 560 || Loss: 5.5884 || 10iter: 10.4459 sec.
-------------
epoch 7 || Epoch_TRAIN_Loss:481.8631 ||Epoch_VAL_Loss:0.0000
timer:  86.2150 sec.
-------------
Epoch 8/10
-------------
（train）
イテレーション 570 || Loss: 5.7912 || 10iter: 3.1521 sec.
イテレーション 580 || Loss: 5.5514 || 10iter: 10.9744 sec.
イテレーション 590 || Loss: 5.3056 || 10iter: 10.3595 sec.
イテレーション 600 || Loss: 6.4106 || 10iter: 10.1458 sec.
イテレーション 610 || Loss: 6.0081 || 10iter: 10.5076 sec.
イテレーション 620 || Loss: 5.2801 || 10iter: 11.1667 sec.
イテレーション 630 || Loss: 5.5157 || 10iter: 11.8299 sec.
イテレーション 640 || Loss: 5.6975 || 10iter: 10.8596 sec.
-------------
epoch 8 || Epoch_TRAIN_Loss:479.3660 ||Epoch_VAL_Loss:0.0000
timer:  87.6975 sec.
-------------
Epoch 9/10
-------------
（train）
イテレーション 650 || Loss: 5.8446 || 10iter: 2.0421 sec.
イテレーション 660 || Loss: 6.2351 || 10iter: 10.5426 sec.
イテレーション 670 || Loss: 5.5395 || 10iter: 10.9755 sec.
イテレーション 680 || Loss: 5.7710 || 10iter: 10.8948 sec.
イテレーション 690 || Loss: 5.7952 || 10iter: 10.7570 sec.
イテレーション 700 || Loss: 5.7771 || 10iter: 11.0148 sec.
イテレーション 710 || Loss: 4.8414 || 10iter: 10.7902 sec.
イテレーション 720 || Loss: 4.9342 || 10iter: 11.7566 sec.
-------------
epoch 9 || Epoch_TRAIN_Loss:479.0972 ||Epoch_VAL_Loss:0.0000
timer:  88.2646 sec.
-------------
Epoch 10/10
-------------
（train）
イテレーション 730 || Loss: 6.1888 || 10iter: 1.2385 sec.
イテレーション 740 || Loss: 6.9216 || 10iter: 10.7274 sec.
イテレーション 750 || Loss: 5.6825 || 10iter: 11.2405 sec.
イテレーション 760 || Loss: 5.7843 || 10iter: 10.9219 sec.
イテレーション 770 || Loss: 7.1822 || 10iter: 10.8547 sec.
イテレーション 780 || Loss: 5.4550 || 10iter: 10.9718 sec.
イテレーション 790 || Loss: 5.7807 || 10iter: 11.1340 sec.
イテレーション 800 || Loss: 5.9131 || 10iter: 10.4190 sec.
イテレーション 810 || Loss: 6.2317 || 10iter: 10.4083 sec.
-------------
（val）
-------------
epoch 10 || Epoch_TRAIN_Loss:478.3301 ||Epoch_VAL_Loss:167.0799
timer:  106.9465 sec.
-------------
Epoch 11/10
-------------
（train）
イテレーション 820 || Loss: 5.6602 || 10iter: 10.5941 sec.
イテレーション 830 || Loss: 5.4967 || 10iter: 10.9182 sec.
イテレーション 840 || Loss: 6.2640 || 10iter: 10.5625 sec.
イテレーション 850 || Loss: 5.5213 || 10iter: 11.3437 sec.
イテレーション 860 || Loss: 6.0217 || 10iter: 11.3032 sec.
イテレーション 870 || Loss: 6.3170 || 10iter: 11.2664 sec.
イテレーション 880 || Loss: 5.4631 || 10iter: 10.8478 sec.
イテレーション 890 || Loss: 5.9018 || 10iter: 10.0681 sec.
-------------
epoch 11 || Epoch_TRAIN_Loss:476.9831 ||Epoch_VAL_Loss:0.0000
timer:  87.4065 sec.
```