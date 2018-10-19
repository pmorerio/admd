## code for 'Learning with privileged information via adversarial discriminative modality distillation' paper, in ArXiv.

###### Data and checkpoint directories are defined in utils.py  
###### Arguments are described in utils.py - get_arguments().

###### Train Step 1, e.g.   
depth: python s1_train_stream.py --dset=ntu --modality=depth_bott --eval=cross_subj  
RGB: python s1_train_stream.py --dset=nwucla --modality=rgb  

###### Evaluate twostream model, rgb and depth e.g.  
define the right path for depth and rgb checkpoints inside twostream_depth_rgb.py  
python twostream_depth_rgb --dset=uwa3dii --just_eval

###### Train Hallucination GAN, e.g.
python s2_gan_hall.py --dset=nwucla --ckpt=./checkpoint/nwucla/s1_train_depth_bott_20180101_010101__dset_nwucla_eval_mode_cross_view/model.ckpt  

###### Evaluate twostream model, rgb and hall e.g.  
Define the right path for hall and rgb checkpoints inside twostream_hall_rgb.py  
python twostream_hall_rgb --dset=uwa3dii --just_eval
