import os

def delete_not_found_path(txt_file_path):
    path_list=[]
    with open(txt_file_path, 'r') as file:
        for line in file:
            path_list.append(line.strip())

    new_path_list=[]

    for file_path in path_list:
        if os.path.exists(file_path):
            new_path_list.append(file_path)
        else:
            print(f"{file_path} は存在しません")

    with open(txt_file_path, 'w') as f:
        for item in new_path_list:
            f.write(item + '\n')

delete_not_found_path("/home/ubuntu/Chipathon/train/ssd/data/COCO_data/train.txt")
delete_not_found_path("/home/ubuntu/Chipathon/train/ssd/data/COCO_data/val.txt")
