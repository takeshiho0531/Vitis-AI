def collect_file_path(txt_file_path):
    path_list=[]
    with open(txt_file_path, 'r') as file:
        for line in file:
            # 各行に対する処理を行います
            path_list.append(line.strip())

    new_path_list=[]
    for file in path_list:
        file="/home/ubuntu/Chipathon/train/ssd/data/"+file[31:]
        new_path_list.append(file)

    with open(txt_file_path, 'w') as f:
        for item in new_path_list:
            f.write(item + '\n')


collect_file_path("/home/ubuntu/Chipathon/train/ssd/data/COCO_data/train.txt")
collect_file_path("/home/ubuntu/Chipathon/train/ssd/data/COCO_data/val.txt")