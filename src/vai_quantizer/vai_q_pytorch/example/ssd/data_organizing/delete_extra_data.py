import os


def delete_extra_data(image_dir, txt_dir): # TODO: 同じでいいでしょ
    # 画像ファイルとテキストファイルの拡張子
    image_ext = '.jpg'
    txt_ext = '.txt'

    # ディレクトリ内のすべてのファイルを走査
    txt_path_list=[]
    image_path_list=[]
    for filename in os.listdir(image_dir):
        if filename.endswith(image_ext):
            image_path = os.path.join(image_dir, filename)
            txt_path = os.path.join(txt_dir, filename.replace(image_ext, txt_ext))

            txt_path_list.append(txt_path)
            image_path_list.append(image_path)

            # ペアが存在しない場合は片方のファイルを削除
            if not os.path.exists(txt_path):
                os.remove(image_path)
                print(image_path)
            elif not os.path.exists(image_path):
                os.remove(txt_path)
                print(txt_path)

delete_extra_data(image_dir = '/home/ubuntu/Chipathon/train/ssd/data/COCO_data/test2017/', txt_dir = '/home/ubuntu/Chipathon/train/ssd/data/COCO_data/test2017/')
delete_extra_data(image_dir = '/home/ubuntu/Chipathon/train/ssd/data/COCO_data/val2017/', txt_dir = '/home/ubuntu/Chipathon/train/ssd/data/COCO_data/val2017/')
delete_extra_data(image_dir = '/home/ubuntu/Chipathon/train/ssd/data/COCO_data/train2017/', txt_dir = '/home/ubuntu/Chipathon/train/ssd/data/COCO_data/train2017/')