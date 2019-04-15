import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import sys
sys.path.append('./FCRN-DepthPrediction/tensorflow')

from model import MultiModal


def load_NYUD(image_dir random_seed=None):
	print ('Loading NYUD dataset')
	
	#~ RGB_MEAN = [123.68, 116.779, 103.939]
	splits = ['train','test']
	n_samples = {'train':2186, 'test':2401}
	dataset = {}
	
	for split in splits:
	    print(split)
	    
	    rgb_classes = sorted(glob.glob(image_dir + '/' + split + '/images/*') )
	    depth_classes = sorted(glob.glob(image_dir + '/' + split + '/depth/*') )
	    assert len(rgb_classes) == len(depth_classes)
	    assert len(rgb_classes) == model.no_classes
	    rgb_images = np.zeros((n_samples[split],224,224,3))
	    depth_images = np.zeros((n_samples[split],224,224,3))
	    labels = np.zeros((n_samples[split],1))
	    
	    l = 0
	    c = 0
	
	    for rgb_class_path, depth_class_path in zip(rgb_classes, depth_classes):
		
		rgb_images_list = sorted(glob.glob(rgb_class_path+'/*'))
		depth_images_list = sorted(glob.glob(depth_class_path+'/*'))
		assert len(rgb_images_list) == len(depth_images_list)
		#~ #print str(l)+'/'+str(len(obj_categories))
		
		for rgb_image, depth_image in zip(rgb_images_list, depth_images_list):
		    
		    img = Image.open(rgb_image)
		    img = img.resize((224,224), Image.ANTIALIAS)
		    img = np.array(img, dtype=float) 
		    img[:,:,0] -= RGB_MEAN[0]
		    img[:,:,1] -= RGB_MEAN[1]
		    img[:,:,2] -= RGB_MEAN[2]
		    img = np.expand_dims(img, axis=0) 
		    rgb_images[c] = img
		    
		    # same processing for HHA-encoded images
		    img = Image.open(depth_image)
		    img = img.resize((224,224), Image.ANTIALIAS)
		    img = np.array(img, dtype=float) 
		    img[:,:,0] -= RGB_MEAN[0]
		    img[:,:,1] -= RGB_MEAN[1]
		    img[:,:,2] -= RGB_MEAN[2]
		    img = np.expand_dims(img, axis=0) 
		    depth_images[c] = img
		    
		    labels[c] = l
			
		    c+=1
		    
		l+=1
	    
	    rnd_indices = np.arange(len(labels))
	    if random_seed is not None:
            np.random.seed(random_seed)
            np.random.shuffle(rnd_indices)
            rgb_images = rgb_images[rnd_indices]
            depth_images = depth_images[rnd_indices]
            labels = labels[rnd_indices]
            
	    dataset[split] = {'rgb_images': rgb_images, 'depth_images': depth_images, 'labels': np.squeeze(labels) }
	    
	print('Loaded!')
    return dataset

def predict(model_data_path, image_folder):
    
    dataset = load_NYUD(image_folder)
    
    # Default input size
    height = 224
    width = 224
    channels = 3
    batch_size = 1
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        
        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node:solver.dataset['train']['rgb_images'][0:1]})
        

        

        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_folder', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    predict(args.model_path, image_folder)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



