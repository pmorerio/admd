import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import numpy as np


class MultiModal(object):

	def __init__(self, mode, learning_rate=0.0001):
		
		self.mode = mode
		self.learning_rate = learning_rate
		self.hidden_repr_size = 128
		self.no_classes = 19
        
        
	def modDrop(self, layer, is_training, p_mod = .9, keep_prob = .8):
		'''
		As in Neverova et al. 'ModDrop': std dropout + modality dropping on the input
		'''
		layer = slim.dropout(layer, keep_prob = keep_prob, is_training = is_training)
		on = tf.cast(tf.random_uniform([1]) - p_mod < 0, tf.float32)
		return tf.cond(is_training, lambda: on*layer, lambda: layer)   

        
	def single_stream(self, images, modality, is_training, reuse=False):
		
		with tf.variable_scope(modality, reuse=reuse):
			with slim.arg_scope(resnet_v1.resnet_arg_scope()):
				_, end_points = resnet_v1.resnet_v1_50(images, self.no_classes, is_training=is_training, reuse=reuse)
	    
		net = end_points[modality+'/resnet_v1_50/block4'] #last bottleneck before logits
		if 'autoencoder' in self.mode:
			return net
		
		with tf.variable_scope(modality+'/resnet_v1_50', reuse=reuse):
			bottleneck = slim.conv2d(net, self.hidden_repr_size , [7, 7], padding='VALID', activation_fn=tf.nn.relu, scope='f_repr')
			net = slim.conv2d(bottleneck, self.no_classes , [1,1], activation_fn=None, scope='_logits_')
			
		if ('train_hallucination' in self.mode or 'test_disc' in self.mode):
			return net, bottleneck
			
		return net


	def D(self, features, reuse=False):
		with tf.variable_scope('discriminator',reuse=reuse):
			with slim.arg_scope([slim.fully_connected],weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.0)):		
				net = slim.fully_connected(features, 1024, activation_fn = tf.nn.relu, scope='disc_fc1')
				#~ if self.mode == 'train_hallucination_p2':
				res = slim.fully_connected(net, 1024, activation_fn = None, scope='disc_res1')
				net = tf.nn.relu(res+net)
				res = slim.fully_connected(net, 1024, activation_fn = None, scope='disc_res2')				
				net = tf.nn.relu(res+net)
				net = slim.fully_connected(net, 2048, activation_fn = tf.nn.relu, scope='disc_fc2')
				net = slim.fully_connected(net, 3076, activation_fn = tf.nn.relu, scope='disc_fc3')
				if self.mode == 'train_hallucination_p2':
					net = slim.fully_connected(net,self.no_classes+1,activation_fn=None,scope='disc_prob')
				elif self.mode == 'train_hallucination':
					net = slim.fully_connected(net,1,activation_fn=tf.sigmoid,scope='disc_prob')
				else:
					print('Unrecognized mode')
		return net
	
	
	def decoder(self, features, is_training, reuse=False):
		# input features from the resnet should be (batch_size, 7, 7, 2048)
		with tf.variable_scope('decoder', reuse=reuse):
				with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,           
									 stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
					with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
										 activation_fn=tf.nn.relu, is_training=is_training):

						net = slim.conv2d_transpose(features, 1024, [3, 3], scope='conv_transpose1')   # (batch_size, 14, 14, channels)
						net = slim.batch_norm(net, scope='bn1')
						net = slim.conv2d_transpose(net, 512, [3, 3], scope='conv_transpose2')  # (batch_size, 28, 28, channels)
						net = slim.batch_norm(net, scope='bn2')
						net = slim.conv2d_transpose(net, 256, [5, 5], scope='conv_transpose3')  # (batch_size, 56, 56, channels)
						net = slim.batch_norm(net, scope='bn3')
						net = slim.conv2d_transpose(net, 128, [5, 5], scope='conv_transpose4')  # (batch_size, 112, 112, channels)
						net = slim.batch_norm(net, scope='bn4')
						net = slim.conv2d_transpose(net, 3, [5, 5],activation_fn=tf.nn.tanh, scope='conv_transpose_out')  # (batch_size, 224, 224, 3)
						
						#normalize output
						RGB_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, name='rgb_mean')
						net = 255*net - RGB_MEAN
						
		return net


	def build_model(self):
		
		if '_rgb' in self.mode  or  '_depth' in self.mode:
            
			modality = self.mode.split('_')[-1]
			self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], modality+'_images')
			self.labels = tf.placeholder(tf.int64, [None], 'labels')
			self.is_training = tf.placeholder(tf.bool, name='is_training')
			self.logits = self.single_stream(self.images, modality=modality, is_training=self.is_training)
			self.pred = tf.argmax(tf.squeeze(self.logits), 1) 
			self.correct_pred = tf.equal(self.pred, self.labels) 
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
			
			if 'train_' in self.mode:
				# training stuff
				t_vars = tf.trainable_variables()
				train_vars = t_vars 
				self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=tf.one_hot(self.labels,self.no_classes )))
				gradients = tf.gradients(self.loss, train_vars)
				gradients = list(zip(gradients, train_vars))
				self.optimizer = tf.train.AdamOptimizer(self.learning_rate)	    
				#~ self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
				self.train_op = self.optimizer.apply_gradients(grads_and_vars=gradients)

				# summary stuff
				loss_summary = tf.summary.scalar('classification_loss', self.loss)
				accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
				self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])


		elif 'train_double_stream' in self.mode:
			
			self.depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'depth_images')
			self.rgb_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'rgb_images')
			self.labels = tf.placeholder(tf.int64, [None], 'labels')
			self.is_training = tf.placeholder(tf.bool, name='is_training')
			
			if self.mode == 'train_double_stream_moddrop':
				self.depth_images = self.modDrop(self.depth_images, is_training=self.is_training)
				self.rgb_images = self.modDrop(self.rgb_images, is_training=self.is_training)

			self.depth_logits = self.single_stream(self.depth_images, modality='depth', is_training=self.is_training)
			self.rgb_logits = self.single_stream(self.rgb_images, modality='rgb', is_training=self.is_training)
			self.logits = (self.depth_logits + self.rgb_logits)/2.
			self.pred = tf.argmax(tf.squeeze(self.logits), 1) 
			self.correct_pred = tf.equal(self.pred, self.labels) 
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

			# training stuff
			t_vars = tf.trainable_variables()
			train_vars = t_vars 
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=tf.one_hot(self.labels,self.no_classes )))
			gradients = tf.gradients(self.loss, train_vars)
			gradients = list(zip(gradients, train_vars))
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate)	    
			#~ self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
			self.train_op = self.optimizer.apply_gradients(grads_and_vars=gradients)

			# summary stuff
			loss_summary = tf.summary.scalar('classification_loss', self.loss)
			accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
			self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])
			
			
		elif self.mode == 'test_ensemble_baseline':
			
			self.depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'depth_images')  ## not used, just to recycle eval function
			self.rgb_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'rgb_images')
			self.labels = tf.placeholder(tf.int64, [None], 'labels')
			self.is_training = tf.placeholder(tf.bool, name='is_training')

			self.rgb1_logits = self.single_stream(self.rgb_images, modality='rgb1', is_training=self.is_training)
			self.rgb_logits = self.single_stream(self.rgb_images, modality='rgb', is_training=self.is_training)
			self.logits = (self.rgb1_logits + self.rgb_logits)/2.
			self.pred = tf.argmax(tf.squeeze(self.logits), 1) 
			self.correct_pred = tf.equal(self.pred, self.labels) 
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
	    
	    
		elif 'train_hallucination' in self.mode:
	    
			#depth & hall streams
			self.depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'depth_images')
			self.rgb_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'rgb_images')
			self.labels = tf.placeholder(tf.int64, [None], 'labels')
			self.is_training = tf.placeholder(tf.bool, name='is_training')
			self.depth_logits, self.depth_features = self.single_stream(self.depth_images, modality='depth', is_training=self.is_training)
			self.hall_logits, self.hall_features = self.single_stream(self.rgb_images, modality='hall', is_training=self.is_training)
		
			#overall acc_hall
			self.pred = tf.argmax(tf.squeeze(self.hall_logits), 1) 
			self.correct_pred = tf.equal(self.pred, self.labels) 
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
			#~ #hall_acc
			#~ self.hall_pred = tf.argmax(tf.squeeze(self.hall_logits), 1) 
			#~ self.hall_correct_pred = tf.equal(self.hall_pred, self.labels) 
			#~ self.hall_accuracy = tf.reduce_mean(tf.cast(self.hall_correct_pred, tf.float32))
			#~ #depth_acc
			#~ self.depth_pred = tf.argmax(tf.squeeze(self.depth_logits), 1) 
			#~ self.depth_correct_pred = tf.equal(self.depth_pred, self.labels) 
			#~ self.depth_accuracy = tf.reduce_mean(tf.cast(self.depth_correct_pred, tf.float32))
			
			#discriminator
			self.logits_real = self.D(self.depth_features,reuse=False)
			self.logits_fake = self.D(self.hall_features, reuse=True )

			#losses
			if self.mode == 'train_hallucination':
				self.d_loss_real = tf.reduce_mean(tf.square(self.logits_real - tf.ones_like(self.logits_real)))
				self.d_loss_fake = tf.reduce_mean(tf.square(self.logits_fake - tf.zeros_like(self.logits_fake)))
				self.d_loss = self.d_loss_real + self.d_loss_fake
				self.g_loss = tf.reduce_mean(tf.square(self.logits_fake - tf.ones_like(self.logits_fake)))
				
				#~ self.d_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
				#~ self.g_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
			
			elif self.mode == 'train_hallucination_p2':
				fake_labels = self.labels + self.no_classes - self.labels ## the last class is the fake one
				self.d_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_real, 
																			labels=tf.one_hot(self.labels,self.no_classes+1) ) )
				self.d_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_fake, 
																			labels=tf.one_hot(fake_labels,self.no_classes+1) ) )
				self.d_loss = self.d_loss_real + self.d_loss_fake
				self.g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_fake,
																			labels=tf.one_hot(self.labels,self.no_classes+1) ) )
			else:
				print('Error building model')
				
			self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate)
			self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate)
			
			t_vars = tf.trainable_variables()
			d_vars = [var for var in t_vars if 'discriminator' in var.name]
			g_vars = [var for var in t_vars if 'hall' in var.name]
			# train ops
			with tf.variable_scope('train_op',reuse=False):
				self.d_train_op = slim.learning.create_train_op(self.d_loss, self.d_optimizer, variables_to_train=d_vars)
				self.g_train_op = slim.learning.create_train_op(self.g_loss, self.g_optimizer, variables_to_train=g_vars)

			#summaries
			d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
			g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
			#hall_acc_summary = tf.summary.scalar('hall_acc', self.accuracy)
			self.summary_op = tf.summary.merge([d_loss_summary, g_loss_summary])


		elif self.mode == 'finetune_hallucination':
	    
			#depth & hall streams
			self.depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'depth_images')  ## not used, just to recycle eval function
			self.rgb_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'rgb_images')
			self.labels = tf.placeholder(tf.int64, [None], 'labels')
			self.is_training = tf.placeholder(tf.bool, name='is_training')
			self.rgb_logits  = self.single_stream(self.rgb_images, modality='rgb', is_training=self.is_training)
			self.hall_logits = self.single_stream(self.rgb_images, modality='hall', is_training=self.is_training)
			self.logits = (self.rgb_logits + self.hall_logits)/2.
			
			#overall acc_hall
			self.pred = tf.argmax(tf.squeeze(self.logits), 1) 
			self.correct_pred = tf.equal(self.pred, self.labels) 
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
			#~ #hall_acc
			#~ self.hall_pred = tf.argmax(tf.squeeze(self.hall_logits), 1) 
			#~ self.hall_correct_pred = tf.equal(self.hall_pred, self.labels) 
			#~ self.hall_accuracy = tf.reduce_mean(tf.cast(self.hall_correct_pred, tf.float32))
			#~ #rgb_acc
			#~ self.rgb_pred = tf.argmax(tf.squeeze(self.rgb_logits), 1) 
			#~ self.rgb_correct_pred = tf.equal(self.rgb_pred, self.labels) 
			#~ self.rgb_accuracy = tf.reduce_mean(tf.cast(self.rgb_correct_pred, tf.float32))
			
			# training stuff
			t_vars = tf.trainable_variables()
			train_vars = t_vars 
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=tf.one_hot(self.labels,self.no_classes )))
			gradients = tf.gradients(self.loss, train_vars)
			gradients = list(zip(gradients, train_vars))
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate)	    
			#~ self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
			self.train_op = self.optimizer.apply_gradients(grads_and_vars=gradients)

			# summary stuff
			loss_summary = tf.summary.scalar('classification_loss', self.loss)
			accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
			self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])


		elif self.mode == 'test_moddrop':
	    
			#rgb & blank depth streams
			self.depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'depth_images') 
			self.blank_depth = self.depth_images - self.depth_images ## bad trick to blank out depth....
			self.rgb_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'rgb_images')
			self.labels = tf.placeholder(tf.int64, [None], 'labels')
			self.is_training = tf.placeholder(tf.bool, name='is_training')
			self.rgb_logits = self.single_stream(self.rgb_images, modality='rgb', is_training=self.is_training)
			#swap between the two
			self.depth_logits = self.single_stream(self.depth_images, modality='depth', is_training=self.is_training)
			#~ self.depth_logits = self.single_stream(self.blank_depth, modality='depth', is_training=self.is_training)

			#overall acc
			# swap between the two 
			self.logits = (self.rgb_logits + self.depth_logits)/2.
			#~ self.logits = self.rgb_logits
			self.pred = tf.argmax(tf.squeeze(self.logits), 1) 
			self.correct_pred = tf.equal(self.pred, self.labels) 
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
			
			
		elif self.mode == 'test_hallucination':
	    
			#rgb & hall streams
			self.depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'depth_images') ## not used, just to recycle eval function
			self.rgb_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'rgb_images')
			self.labels = tf.placeholder(tf.int64, [None], 'labels')
			self.is_training = tf.placeholder(tf.bool, name='is_training')
			self.rgb_logits = self.single_stream(self.rgb_images, modality='rgb', is_training=self.is_training)
			self.hall_logits = self.single_stream(self.rgb_images, modality='hall', is_training=self.is_training)

			#overall acc
			self.logits = (self.rgb_logits + self.hall_logits)/2.
			self.pred = tf.argmax(tf.squeeze(self.logits), 1) 
			self.correct_pred = tf.equal(self.pred, self.labels) 
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
			#hall_acc
			#~ self.hall_pred = tf.argmax(tf.squeeze(self.hall_logits), 1) 
			#~ self.hall_correct_pred = tf.equal(self.hall_pred, self.labels) 
			#~ self.hall_accuracy = tf.reduce_mean(tf.cast(self.hall_correct_pred, tf.float32))
			#~ #rgb_acc
			#~ self.rgb_pred = tf.argmax(tf.squeeze(self.rgb_logits), 1) 
			#~ self.rgb_correct_pred = tf.equal(self.rgb_pred, self.labels) 
			#~ self.rgb_accuracy = tf.reduce_mean(tf.cast(self.rgb_correct_pred, tf.float32))
		
		
		elif self.mode == 'train_autoencoder' or self.mode == 'test_autoencoder' :
			
			self.depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'depth_images') 
			self.rgb_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'rgb_images')
			self.is_training = tf.placeholder(tf.bool, name='is_training')
			self.rgb_features = self.single_stream(self.rgb_images, modality='rgb', is_training=self.is_training)
			self.reconstructed_depth = self.decoder(self.rgb_features, is_training=self.is_training)
			
			self.loss = tf.reduce_mean(tf.square(self.depth_images-self.reconstructed_depth))
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate)	 
			self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)

			loss_summary = tf.summary.scalar('reconstruction_loss', self.loss)
			rec_depth_summary = tf.summary.image('reconstructed', self.reconstructed_depth)
			depth_image_summary = tf.summary.image('depth', self.depth_images)
			self.summary_op = tf.summary.merge([loss_summary, rec_depth_summary, depth_image_summary])


		elif 'test_double_stream' in self.mode:

			self.depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'depth_images') #to load precomputed reconstructed images
			self.rgb_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'rgb_images')
			self.is_training = tf.placeholder(tf.bool, name='is_training')
			self.labels = tf.placeholder(tf.int64, [None], 'labels')
			
			self.depth_logits = self.single_stream(self.depth_images, modality='depth', is_training=self.is_training)
			self.rgb_logits = self.single_stream(self.rgb_images, modality='rgb', is_training=self.is_training)
			self.logits = (self.depth_logits + self.rgb_logits)/2.
			self.pred = tf.argmax(tf.squeeze(self.logits), 1) 
			self.correct_pred = tf.equal(self.pred, self.labels) 
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
			
			
		elif self.mode == 'test_disc':
			
			#depth & rgb streams
			self.depth_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'depth_images') 
			self.rgb_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'rgb_images')
			self.labels = tf.placeholder(tf.int64, [None], 'labels')
			self.is_training = tf.placeholder(tf.bool, name='is_training')
			self.rgb_logits, self.rgb_features = self.single_stream(self.rgb_images, modality='rgb', is_training=self.is_training)
			self.depth_logits, self.depth_features = self.single_stream(self.depth_images, modality='depth', is_training=self.is_training)
		
			#overall acc_hall
			self.logits = (self.rgb_logits + self.hall_logits)/2.
			self.pred = tf.argmax(tf.squeeze(self.logits), 1) 
			self.correct_pred = tf.equal(self.pred, self.labels) 
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
			#depth_acc
			self.depth_pred = tf.argmax(tf.squeeze(self.depth_logits), 1) 
			self.depth_correct_pred = tf.equal(self.depth_pred, self.labels) 
			self.depth_accuracy = tf.reduce_mean(tf.cast(self.depth_correct_pred, tf.float32))
			#rgb_acc
			self.rgb_pred = tf.argmax(tf.squeeze(self.rgb_logits), 1) 
			self.rgb_correct_pred = tf.equal(self.rgb_pred, self.labels) 
			self.rgb_accuracy = tf.reduce_mean(tf.cast(self.rgb_correct_pred, tf.float32))
			
			#discriminator
			self.logits_preds = tf.nn.softmax(self.D(self.rgb_features,reuse=False))
			self.logits_preds = tf.nn.softmax(self.D(self.depth_features, reuse=True ))	
	
		
		
		
		
		
		
		
		
		
		
		
		
		
		
