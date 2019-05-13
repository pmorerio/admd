#!/bin/bash
python main.py --bs 64 --lr 1e-4 --mode train_rgb --it 2000 > rgb1.txt                                
python codebase_rename_ckpt.py --checkpoint_dir=./model/rgb --checkpoint_new=model/rgb1  --replace_from='rgb/resnet_v1_50/'  --replace_to='rgb1/resnet_v1_50/'
python main.py --mode test_rgb1
python main.py --bs 64 --lr 1e-4 --mode train_rgb --it 2000 > rgb.txt
python main.py --mode test_rgb
python main.py --mode test_ensemble_baseline
python main.py --bs 64 --lr 1e-4 --mode train_depth --it 2000 > depth.txt
python main.py --mode test_depth

# just to save in one single model and to eval on test set
# finetuning does not seem to help much and sometimes it even lowers accuracy
# need very low lr
python main.py --bs 32 --lr 1e-6 --mode train_double_stream --it 1000 > double.txt 

python main.py --bs 64 --lr 1e-6 --mode train_hallucination_p2 --it 5000 > hall_p2.txt
python main.py  --mode test_hallucination
 
# std adversarial game
python main.py --bs 64 --lr 1e-6 --mode train_hallucination --it 10000 > hall.txt
python main.py  --mode test_hallucination

python main.py --bs 32 --lr 1e-5 --mode train_double_stream_moddrop  --it 1000 > moddrop.txt


python main.py --bs 32 --lr 1e-4 --mode train_autoencoder --it 10000
