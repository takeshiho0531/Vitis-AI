import numpy as np
import cv2
from tqdm import tqdm

def get_file_path(txt_file_path):
    path_list=[]
    with open(txt_file_path, 'r') as file:
        for line in file:
            # 各行に対する処理を行います
            path_list.append(line.strip())
    return path_list


file_paths=get_file_path("/home/ubuntu/Chipathon/train/ssd/data/COCO_data/train.txt")

bgr_sum = np.zeros(3)  # BGRの合計を初期化
num_images = len(file_paths)  # 画像の数を取得

for file_path in tqdm(file_paths):
    image = cv2.imread(file_path)
    bgr_sum += np.sum(image, axis=(0, 1))  # BGRの合計を加算

bgr_mean = bgr_mean = bgr_sum / (num_images * image.shape[0] * image.shape[1] * 256) # 合計を画像の総ピクセル数で割って平均を計算

print("bgr_mean",bgr_mean)