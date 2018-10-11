#!/bin/bash
#rename resnet weights and checkpoints
python codebase_rename_ckpt.py --checkpoint_dir=/data/models/resnet_50/resnet_v1_50.ckpt --checkpoint_new=/data/models/resnet_50/resnet_v1_50_depth.ckpt  --replace_from='resnet_v1_50/'  --replace_to='depth/resnet_v1_50/' 
python codebase_rename_ckpt.py --checkpoint_dir=/data/models/resnet_50/resnet_v1_50.ckpt --checkpoint_new=/data/models/resnet_50/resnet_v1_50_rgb.ckpt  --replace_from='resnet_v1_50/'  --replace_to='rgb/resnet_v1_50/' 

# this is needed to train two different rgb models for the ensemble baseline 
#~ python codebase_rename_ckpt.py --checkpoint_dir=./model/rgb --checkpoint_new=model/rgb1  --replace_from='rgb/resnet_v1_50/'  --replace_to='rgb1/resnet_v1_50/'
