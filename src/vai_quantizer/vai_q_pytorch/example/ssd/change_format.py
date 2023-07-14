import os

# ディレクトリのパス
directory = "/home/ubuntu/Chipathon/train/ssd/data/COCO_data"

# ディレクトリ内のすべてのファイルのパスを取得
file_paths = []
for root, directories, files in os.walk(directory):
    for file in files:
        if file.endswith('.txt'):  # .txt拡張子のファイルのみを対象とする場合
            file_paths.append(os.path.join(root, file))

# ファイルパスのリストを表示
print(file_paths)


def change_format(path):
  result = []

  with open(path, 'r') as file:
    for line in file:
        line = line.strip()
        values = [(float(num)) for num in line.split()]
        result.append(values)

  for each_result in result:
    each_result_copy=each_result.copy()
    each_result[0]=each_result_copy[1]-0.5*each_result_copy[3]
    each_result[1]=each_result_copy[2]-0.5*each_result_copy[4]
    each_result[2]=each_result_copy[1]+0.5*each_result_copy[3]
    each_result[3]=each_result_copy[2]+0.5*each_result_copy[4]
    each_result[4]=each_result_copy[0]

  with open(path, 'w') as file:
    for values in result:
        line = ' '.join(str(num) for num in values) + '\n'
        file.write(line)
  print("changed", path)


for file_path in file_paths:
  change_format(file_path)
print(len(file_paths))
