import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import glob
import sys
sys.path.append('./FCRN-DepthPrediction/tensorflow')

import models
#~ from tifffile import imsave
#~ from scipy.misc import imsave
#~ from libtiff import TIFF
import cv2

def predict(model_data_path, image_dir):
    
    # Default input size


    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, 448, 448, 3))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, 1, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)
    
	print ('Processing NYUD dataset')
	
	#~ RGB_MEAN = [123.68, 116.779, 103.939]
	splits = ['train','test']
	n_samples = {'train':2186, 'test':2401}
	
	for split in splits:
	    print('[*] '+split)
	    
	    rgb_classes = sorted(glob.glob(image_dir + '/' + split + '/images/*') )

	    assert len(rgb_classes) == 19
	    
	    l = 0
	    c = 0
	
	    for rgb_class_path in rgb_classes:
		
		class_dir = rgb_class_path.split('/')[-1]
		print(class_dir)
		save_dir = image_dir + '/' + split + '/FCRN_depth/' + class_dir
		if not os.path.exists(save_dir):
		    os.makedirs(save_dir)
		    
		rgb_images_list = sorted(glob.glob(rgb_class_path+'/*'))
		
		for rgb_image in rgb_images_list:
		    image_name = rgb_image.split('/')[-1]
		    img = Image.open(rgb_image)
		    img = img.resize((448,448), Image.ANTIALIAS)
		    img = np.array(img, dtype=float) 
		    img = np.expand_dims(img, axis=0) 
		    
		    # Evalute the network for the given image
		    pred  = sess.run(net.get_output(), feed_dict={input_node:img})
		    pred = np.squeeze(pred, axis=0)
		    max_val = np.max(pred)
		    min_val = np.min(pred)
		    pred = (pred - min_val)*255. / (max_val - min_val)
		    pred = cv2.applyColorMap(pred.astype('uint8'), cv2.COLORMAP_JET)
		    cv2.imwrite(os.path.join(save_dir, image_name), pred)
		    
		    
		    #~ tiff = TIFF.open(os.path.join(save_dir, image_name), mode='w')
		    #~ tiff.write_image(pred)
		    #~ tiff.close()
		    
		    #~ imsave(os.path.join(save_dir, image_name), pred)
		    #~ pred = Image.fromarray(pred.astype('uint16'), mode='I')
		    #~ print(pred)
		    #~ pred.save(os.path.join(save_dir, image_name + '.png'), bits=16)
		    #~ imsave(os.path.join(save_dir, image_name + '.png'))
		    #~ np.save(os.path.join(save_dir, image_name), np.squeeze(pred))
		    
		    #~ fig = plt.figure()
		    #~ ii = plt.imshow(pred)
		    #~ fig.colorbar(ii)
		    #~ plt.show()
		    #~ return 0
		    c+=1
		    
		l+=1
		
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_folder', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    predict(args.model_path, args.image_folder)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



