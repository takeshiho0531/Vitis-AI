# Chipathon/train/ssdをVitis-AI/src/vai_quantizer/vai_q_pytorch/example/以下にcpしてくる
# dataは/workspace/Vitis-AI/src/vai_quantizer/vai_q_pytorch/example/ssd/data/
# pretrained modelは/workspace/Vitis-AI/src/vai_quantizer/vai_q_pytorch/example/ssd/trained/
from ssd.ssd_model import DataTransform, VOCDataset, od_collate_fn, MultiBoxLoss, SSD, make_datapath_list, Anno_txt2list
import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
from pytorch_nndct.apis import Inspector

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()


parser.add_argument(
    '--model_dir',
    default="/workspace/Vitis-AI/src/vai_quantizer/vai_q_pytorch/example/ssd/trained/",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth'
)
parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--inspect', 
    dest='inspect',
    action='store_true',
    help='inspect model')

parser.add_argument('--target', 
    dest='target',
    nargs="?",
    const="",
    help='specify target device')

args, _ = parser.parse_known_args()


def quantization(title='optimize',
                 model_name='',   # TODO
                 file_path=''): 
  quant_mode = args.quant_mode
  deploy = args.deploy
  batch_size = args.batch_size
  subset_len = args.subset_len
  inspect = args.inspect
  config_file = args.config_file
  target = args.target
  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1 or subset_len != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1
    subset_len = 1

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

  model = SSD(phase="train", cfg=ssd_cfg)
  model.cpu()
  model.load_state_dict(torch.load(file_path))

  input = torch.randn([batch_size, 3, 224, 224])
  if quant_mode == 'float':
    quant_model = model
    if inspect:
      if not target:
          raise RuntimeError("A target should be specified for inspector.")

      # create inspector
      inspector = Inspector(target)  # by name
      # start to inspect
      inspector.inspect(quant_model, (input,), device=device)
      sys.exit()

  else:
    ####################################################################################
    # This function call will create a quantizer object and setup it. 
    # Eager mode model code will be converted to graph model. 
    # Quantization is not done here if it needs calibration.
    quantizer = torch_quantizer(
        quant_mode, model, (input), device=device, quant_config_file=config_file, target=target)

    # Get the converted model to be quantized.
    quant_model = quantizer.quant_model
    #####################################################################################


  # handle quantization result
  if quant_mode == 'calib':
    # Exporting intermediate files will be used when quant_mode is 'test'. This is must.
    quantizer.export_quant_config()
  if deploy:
    quantizer.export_torch_script()
    quantizer.export_onnx_model()
    quantizer.export_xmodel()