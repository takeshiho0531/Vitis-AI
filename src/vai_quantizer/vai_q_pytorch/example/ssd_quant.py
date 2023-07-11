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

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()


parser.add_argument(
    '--data_dir',
    default="/workspace/Vitis-AI/src/vai_quantizer/vai_q_pytorch/example/ssd/data/",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
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
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
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



def load_data(train=True,
              data_dir='/workspace/Vitis-AI/src/vai_quantizer/vai_q_pytorch/example/ssd/data/',
              batch_size=32,
              sample_method='random',
              distributed=False,
              **kwargs):

  #prepare data
  # random.seed(12345)
  train_sampler = None

  train_img_list, train_anno_list, val_img_list, val_anno_list=make_datapath_list(train_img_path="/home/ubuntu/Chipathon/train/ssd/data/COCO_data/train.txt", 
                                                                                val_img_path="/home/ubuntu/Chipathon/train/ssd/data/COCO_data/val.txt")
  
  coco_classes = ['person','bicycle', 'car']
  color_mean=(104, 117, 123) #TODO
  input_size = 300

  if train:
    train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
        input_size, color_mean), transform_anno=Anno_txt2list(coco_classes))

    if distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=od_collate_fn)


  else:
    val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
        input_size, color_mean), transform_anno=Anno_txt2list(coco_classes))

    dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=od_collate_fn)
  return data_loader, train_sampler
