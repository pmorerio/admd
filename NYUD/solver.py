import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import glob
from PIL import Image
import cPickle


class Solver(object):

    def __init__(self, model, batch_size=4, train_iter=20000, train_iter_adv=200000, log_dir='logs',
                 model_save_path='model',
                 resnet50_ckpt='/data/models/resnet_50/',
                 image_dir='/data/datasets/NYUD_multimodal'):

        self.model = model
        self.batch_size = batch_size
        self.train_iter = train_iter
        self.train_iter_adv = train_iter_adv
        self.log_dir = os.path.join(log_dir, self.model.mode)
        self.model_save_path = model_save_path

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = False
        self.config.allow_soft_placement = True
        self.resnet50_ckpt = resnet50_ckpt
        self.no_classes = model.no_classes

        self.image_dir = image_dir
        self.load_NYUD()

    def load_NYUD(self):
        print ('Loading NYUD dataset')

        RGB_MEAN = [123.68, 116.779, 103.939]
        splits = ['train', 'test']
        self.n_samples = {'train': 2186, 'test': 2401}
        self.dataset = {}

        for split in splits:
            print(split)

            rgb_classes = sorted(
                glob.glob(self.image_dir + '/' + split + '/images/*'))
            depth_classes = sorted(
                glob.glob(self.image_dir + '/' + split + '/depth/*'))
            assert len(rgb_classes) == len(depth_classes)
            assert len(rgb_classes) == self.model.no_classes
            rgb_images = np.zeros((self.n_samples[split], 224, 224, 3))
            depth_images = np.zeros((self.n_samples[split], 224, 224, 3))
            labels = np.zeros((self.n_samples[split], 1))

            l = 0
            c = 0

            for rgb_class_path, depth_class_path in zip(rgb_classes, depth_classes):

                rgb_images_list = sorted(glob.glob(rgb_class_path + '/*'))
                depth_images_list = sorted(glob.glob(depth_class_path + '/*'))
                assert len(rgb_images_list) == len(depth_images_list)
                # ~ #print str(l)+'/'+str(len(obj_categories))

                for rgb_image, depth_image in zip(rgb_images_list, depth_images_list):

                    img = Image.open(rgb_image)
                    img = img.resize((224, 224), Image.ANTIALIAS)
                    img = np.array(img, dtype=float)
                    img[:, :, 0] -= RGB_MEAN[0]
                    img[:, :, 1] -= RGB_MEAN[1]
                    img[:, :, 2] -= RGB_MEAN[2]
                    img = np.expand_dims(img, axis=0)
                    rgb_images[c] = img

                    # same processing for HHA-encoded images
                    img = Image.open(depth_image)
                    img = img.resize((224, 224), Image.ANTIALIAS)
                    img = np.array(img, dtype=float)
                    img[:, :, 0] -= RGB_MEAN[0]
                    img[:, :, 1] -= RGB_MEAN[1]
                    img[:, :, 2] -= RGB_MEAN[2]
                    img = np.expand_dims(img, axis=0)
                    depth_images[c] = img

                    labels[c] = l

                    c += 1

                l += 1

            rnd_indices = np.arange(len(labels))
            np.random.seed(231)
            np.random.shuffle(rnd_indices)
            rgb_images = rgb_images[rnd_indices]
            depth_images = depth_images[rnd_indices]
            labels = labels[rnd_indices]
            self.dataset[split] = {
                'rgb_images': rgb_images, 'depth_images': depth_images, 'labels': np.squeeze(labels)}

        print('Loaded!')

    def eval_all_single_stream(self, split, modality, session):
        modality = 'rgb' if 'rgb' in modality else modality

        # is_training = False for testing to be fair
        batches_per_epochs = int(
            self.dataset[split]['labels'].shape[0] / self.batch_size) + 1
        #print('Evaluating '+split+' accuracy')
        correct_preds = 0.
        for _im, _lab, in zip(np.array_split(self.dataset[split][modality + '_images'], batches_per_epochs),
                              np.array_split(
            self.dataset[split]['labels'], batches_per_epochs),
        ):
            feed_dict = {self.model.images: _im, self.model.labels: _lab,
                         self.model.is_training: split == 'train'}
            _acc_ = session.run(fetches=self.model.accuracy,
                                feed_dict=feed_dict)
            # must be a weighted average since last split is smaller
            correct_preds += (_acc_ * len(_lab))
        print (modality + ' ' + split
               + ' acc [%.4f]' % (correct_preds / len(self.dataset[split]['labels'])))

    def train_single_stream(self, modality):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()

            print ('---Do not forget to rename the variables in the orginal resnet checkpoint if you are training for the first time')
            print ('---Run rename_ckpt.sh')
            print ('Loading pretrained ' + modality + '/resnet50...')
            variables_to_restore = slim.get_model_variables(
                scope=modality + '/resnet_v1_50')
            # get rid of logits
            variables_to_restore = [
                vv for vv in variables_to_restore if 'logits' not in vv.name]
            variables_to_restore = [
                vv for vv in variables_to_restore if 'f_repr' not in vv.name]
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.resnet50_ckpt +
                             'resnet_v1_50_' + modality + '.ckpt')
            print('Loaded!')

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(
                logdir=self.log_dir, graph=tf.get_default_graph())
            # the +1 gives an additional smaller batch
            batches_per_epochs = int(
                self.dataset['train']['labels'].shape[0] / self.batch_size) + 1

            for step in range(self.train_iter):
                i = step % batches_per_epochs
                feed_dict = {self.model.images: self.dataset['train'][modality + '_images'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.labels: self.dataset['train']['labels'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.is_training: True}

                if step % 10 == 0:
                    summary, l, acc = sess.run(
                        [self.model.summary_op, self.model.loss, self.model.accuracy], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d]  loss: [%.6f] acc: [%.6f] '
                           % (step, self.train_iter, l, acc))

                if i == 0:
                    # Eval on train
                    self.eval_all_single_stream(
                        'train', session=sess, modality=modality)
                    self.eval_all_single_stream(
                        'test', session=sess, modality=modality)
                    saver.save(sess, os.path.join(
                        self.model_save_path, modality))

                sess.run(self.model.train_op, feed_dict)

    def test_single_stream(self, modality):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()

            print ('Loading  ' + modality + '/resnet50...')
            variables_to_restore = slim.get_model_variables(
                scope=modality + '/resnet_v1_50')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, modality))
            print('Loaded!')

            #~ self.eval_all_single_stream('train', session=sess, modality=modality)
            self.eval_all_single_stream(
                'test', session=sess, modality=modality)

    def eval_all_double_stream(self, split, session, noise=0.):
        # is_training = False for testing to be fair
        batches_per_epochs = int(
            self.dataset[split]['labels'].shape[0] / self.batch_size) + 1
        #print('Evaluating '+split+' accuracy')
        correct_preds = 0.
        for rgb_im, depth_im, _lab, in zip(np.array_split(self.dataset[split]['rgb_images'], batches_per_epochs),
                                           np.array_split(
            self.dataset[split]['depth_images'], batches_per_epochs),
            np.array_split(
            self.dataset[split]['labels'], batches_per_epochs),
        ):
            if noise > 0.:
                depth_im = depth_im * \
                    np.random.normal(1, noise, size=depth_im.shape)
            elif noise < 0.:
                depth_im = np.zeros(shape=depth_im.shape)
            feed_dict = {self.model.rgb_images: rgb_im, self.model.depth_images: depth_im,
                         self.model.labels: _lab, self.model.is_training: split == 'train'}
            _acc_ = session.run(fetches=self.model.accuracy,
                                feed_dict=feed_dict)
            # must be a weighted average since last split is smaller
            correct_preds += (_acc_ * len(_lab))
        print (split + ' acc [%.4f]' %
               (correct_preds / len(self.dataset[split]['labels'])))

    def train_double_stream(self):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()

            for modality in ['rgb', 'depth']:
                print ('Loading pretrained ' + modality + '/resnet50...')
                variables_to_restore = slim.get_model_variables(
                    scope=modality + '/resnet_v1_50')
                restorer = tf.train.Saver(variables_to_restore)
                restorer.restore(sess, os.path.join(
                    self.model_save_path, modality))
                print('Loaded!')

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(
                logdir=self.log_dir, graph=tf.get_default_graph())
            # the +1 gives an additional smaller batch
            batches_per_epochs = int(
                self.dataset['train']['labels'].shape[0] / self.batch_size) + 1

            for step in range(self.train_iter):
                i = step % batches_per_epochs
                feed_dict = {self.model.rgb_images: self.dataset['train']['rgb_images'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.depth_images: self.dataset['train']['depth_images'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.labels: self.dataset['train']['labels'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.is_training: True}

                if step % 10 == 0:
                    summary, l, acc = sess.run(
                        [self.model.summary_op, self.model.loss, self.model.accuracy], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d]  loss: [%.6f] acc: [%.6f] '
                           % (step, self.train_iter, l, acc))

                if i == 0:
                    # Eval on train
                    #~ self.eval_all_double_stream('train', session=sess)
                    self.eval_all_double_stream('test', session=sess)
                    model_name = 'double_stream_moddrop' if 'moddrop' in self.model.mode else 'double_stream'
                    saver.save(sess, os.path.join(
                        self.model_save_path, model_name))

                sess.run(self.model.train_op, feed_dict)

    def test_ensemble_baseline(self):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()

            for modality in ['rgb', 'rgb1']:
                print ('Loading pretrained ' + modality + '/resnet50...')
                variables_to_restore = slim.get_model_variables(
                    scope=modality + '/resnet_v1_50')
                restorer = tf.train.Saver(variables_to_restore)
                restorer.restore(sess, os.path.join(
                    self.model_save_path, modality))
                print('Loaded!')

            #~ self.eval_all_double_stream('train', session=sess)
            self.eval_all_double_stream('test', session=sess)

    def train_hallucination(self):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:

            tf.global_variables_initializer().run()
            # load depth only
            print ('Loading pretrained double_stream model...')
            variables_to_restore = slim.get_model_variables(
                scope='depth/resnet_v1_50')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, 'double_stream'))
            print('Loaded!')

            # re-initialize the hall strem with values from the depth
            print ('Copying depth to hallucination...')
            hall_vars = [var for var in tf.global_variables()
                         if 'hall' in var.name]
            depth_vars = [var for var in tf.global_variables()
                          if 'depth' in var.name]
            for hvar, dvar in zip(hall_vars, depth_vars):
                # ~ print('assigning from \t'+dvar.name)
                # ~ print('to \t\t'+hvar.name)
                assign_op = hvar.assign(dvar)
                sess.run(assign_op)
            print('Done!')

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(
                logdir=self.log_dir, graph=tf.get_default_graph())
            # the +1 gives an additional smaller batch
            batches_per_epochs = int(
                self.dataset['train']['labels'].shape[0] / self.batch_size) + 1
            print(batches_per_epochs, 'batches (i.e. iterations) per epoch')

            for step in range(self.train_iter_adv):
                i = step % batches_per_epochs

                feed_dict = {self.model.rgb_images: self.dataset['train']['rgb_images'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.depth_images: self.dataset['train']['depth_images'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.labels: self.dataset['train']['labels'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.is_training: True}
                if i == 0:
                    # ~ if step%500==0:
                    summary, d_loss, g_loss, logits_real, logits_fake = sess.run([self.model.summary_op, self.model.d_loss, self.model.g_loss,
                                                                                  self.model.logits_real, self.model.logits_fake], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d]  d_loss: [%.6f] g_loss: [%.6f] \n\t\t logits_real: [%.6f] logits fake: [%.6f] '
                           % (step, self.train_iter_adv, d_loss, g_loss, np.mean(logits_real), np.mean(logits_fake)))

                # ~ if i==0:
                    # Eval on train
                    #self.eval_all_double_stream('train', session=sess)
                    self.eval_all_double_stream('test', session=sess)
                    saver.save(sess, os.path.join(
                        self.model_save_path, 'hallucination'))

                sess.run(self.model.d_train_op, feed_dict)
                sess.run(self.model.g_train_op, feed_dict)

    def finetune_hallucination(self):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:

            tf.global_variables_initializer().run()
            # load depth only
            print ('Loading pretrained hallucination model...')
            variables_to_restore = slim.get_model_variables(
                scope='hall/resnet_v1_50')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, 'hallucination'))
            print('Loaded!')

            print ('Loading rgb stream from pretrained double_stream model...')
            variables_to_restore = slim.get_model_variables(
                scope='rgb/resnet_v1_50')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, 'double_stream'))
            print('Loaded!')

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(
                logdir=self.log_dir, graph=tf.get_default_graph())
            # the +1 gives an additional smaller batch
            batches_per_epochs = int(
                self.dataset['train']['labels'].shape[0] / self.batch_size) + 1

            for step in range(self.train_iter):
                i = step % batches_per_epochs
                feed_dict = {self.model.rgb_images: self.dataset['train']['rgb_images'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.labels: self.dataset['train']['labels'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.is_training: True}

                if step % 10 == 0:
                    summary, l, acc = sess.run(
                        [self.model.summary_op, self.model.loss, self.model.accuracy], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d]  loss: [%.6f] acc: [%.6f] '
                           % (step, self.train_iter, l, acc))

                if i == 0:
                    # Eval on train
                    #~ self.eval_all_double_stream('train', session=sess)
                    self.eval_all_double_stream('test', session=sess)
                    saver.save(sess, os.path.join(
                        self.model_save_path, 'hallucination_finetuned'))

                sess.run(self.model.train_op, feed_dict)

    def test_hallucination(self):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:

            tf.global_variables_initializer().run()
            # load depth only
            print ('Loading pretrained hallucination model...')
            variables_to_restore = slim.get_model_variables(
                scope='hall/resnet_v1_50')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, 'hallucination'))
            print('Loaded!')

            print ('Loading rgb stream from pretrained double_stream model...')
            variables_to_restore = slim.get_model_variables(
                scope='rgb/resnet_v1_50')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, 'double_stream'))
            print('Loaded!')

            self.eval_all_double_stream('test', session=sess)

    def test_moddrop(self, noise=0.):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:

            tf.global_variables_initializer().run()
            print ('Loading pretrained double_stream_moddrop model...')
            variables_to_restore = slim.get_model_variables(
                scope='rgb/resnet_v1_50')
            variables_to_restore += slim.get_model_variables(
                scope='depth/resnet_v1_50')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, 'double_stream_moddrop'))
            print('Loaded!')

            self.eval_all_double_stream('test', session=sess, noise=noise)

    def train_autoencoder(self):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()

            modality = 'rgb'
            print ('Loading pretrained ' + modality + '/resnet50...')
            variables_to_restore = slim.get_model_variables(
                scope=modality + '/resnet_v1_50')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, modality))
            print('Loaded!')

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(
                logdir=self.log_dir, graph=tf.get_default_graph())
            # the +1 gives an additional smaller batch
            batches_per_epochs = int(
                self.dataset['train']['labels'].shape[0] / self.batch_size) + 1

            for step in range(self.train_iter):
                i = step % batches_per_epochs
                feed_dict = {self.model.rgb_images: self.dataset['train']['rgb_images'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.depth_images: self.dataset['train']['depth_images'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.is_training: True}

                # ~ if i==0:
                if step % 50 == 0:
                    summary, l = sess.run(
                        [self.model.summary_op, self.model.loss], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d]  loss: [%.6f]'
                           % (step, self.train_iter, l))

                if i == 0:
                    saver.save(sess, os.path.join(
                        self.model_save_path, 'autoencoder'))

                sess.run(self.model.train_op, feed_dict)

    def test_autoencoder(self):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()

            print ('Loading pretrained autoencoder...')
            variables_to_restore = slim.get_model_variables()
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, 'autoencoder'))
            print('Loaded!')

            summary_writer = tf.summary.FileWriter(
                logdir=self.log_dir, graph=tf.get_default_graph())
            # the +1 gives an additional smaller batch
            batches_per_epochs = int(
                self.dataset['test']['labels'].shape[0] / self.batch_size) + 1
            # simply visualize on the test set
            # ... and save a copy of the generated images
            generated_depth_images = np.zeros(
                (self.n_samples['test'], 224, 224, 3))

            for step in range(batches_per_epochs):
                i = step
                feed_dict = {self.model.rgb_images: self.dataset['test']['rgb_images'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.depth_images: self.dataset['test']['depth_images'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.is_training: False}

                summary, l, gen_depth = sess.run(
                    [self.model.summary_op, self.model.loss, self.model.reconstructed_depth], feed_dict)
                generated_depth_images[i *
                                       self.batch_size:(i + 1) * self.batch_size] = gen_depth
                summary_writer.add_summary(summary, step)
                print ('Step: [%d/%d]  loss: [%.6f]'
                       % (step, batches_per_epochs, l))

            np.save('generated_depth_images', generated_depth_images)

    def test_double_stream_with_ae(self):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:

            tf.global_variables_initializer().run()
            print ('Loading pretrained double_stream model...')
            variables_to_restore = slim.get_model_variables(
                scope='rgb/resnet_v1_50')
            variables_to_restore += slim.get_model_variables(
                scope='depth/resnet_v1_50')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, 'double_stream'))
            print('Loaded!')

            generated_depth_images = np.load('generated_depth_images.npy')

            # is_training = False for testing to be fair
            split = 'test'
            batches_per_epochs = int(
                self.dataset[split]['labels'].shape[0] / self.batch_size) + 1
            correct_preds = 0.
            for rgb_im, depth_im, _lab, in zip(np.array_split(self.dataset[split]['rgb_images'], batches_per_epochs),
                                               np.array_split(
                generated_depth_images, batches_per_epochs),
                np.array_split(
                self.dataset[split]['labels'], batches_per_epochs),
            ):
                feed_dict = {self.model.rgb_images: rgb_im, self.model.depth_images: depth_im,
                             self.model.labels: _lab, self.model.is_training: split == 'train'}
                _acc_ = sess.run(fetches=self.model.accuracy,
                                 feed_dict=feed_dict)
                # must be a weighted average since last split is smaller
                correct_preds += (_acc_ * len(_lab))
            print (split + ' acc [%.4f]' %
                   (correct_preds / len(self.dataset[split]['labels'])))

    def test_double_stream(self, noise=0.):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:

            tf.global_variables_initializer().run()
            print ('Loading pretrained double_stream model...')
            variables_to_restore = slim.get_model_variables(
                scope='rgb/resnet_v1_50')
            variables_to_restore += slim.get_model_variables(
                scope='depth/resnet_v1_50')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, 'double_stream'))
            print('Loaded!')

            self.eval_all_double_stream('test', session=sess, noise=noise)

    def test_disc(self, noise=0.):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:

            #~ tf.global_variables_initializer().run()
            print ('Loading pretrained double_stream model...')
            variables_to_restore = slim.get_model_variables(
                scope='rgb/resnet_v1_50')
            variables_to_restore += slim.get_model_variables(
                scope='depth/resnet_v1_50')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, 'double_stream'))
            print('Loaded!')

            print ('Loading pretrained discriminator...')
            variables_to_restore = slim.get_model_variables(scope='disc')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, 'hallucination'))
            print('Loaded!')

            self.eval_all_double_stream('test', session=sess, noise=noise)

    def train_eccv(self):

        # build a graph
        self.model.build_model()

        with tf.Session(config=self.config) as sess:

            tf.global_variables_initializer().run()
            # load depth only
            print ('Loading pretrained double_stream model...')
            variables_to_restore = slim.get_model_variables(
                scope='depth/resnet_v1_50')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                self.model_save_path, 'double_stream'))
            print('Loaded!')

            # re-initialize the hall stream with values from the depth
            print ('Copying depth to hallucination...')
            hall_vars = [var for var in tf.global_variables()
                         if 'hall' in var.name]
            depth_vars = [var for var in tf.global_variables()
                          if 'depth' in var.name]
            for hvar, dvar in zip(hall_vars, depth_vars):
                # ~ print('assigning from \t'+dvar.name)
                # ~ print('to \t\t'+hvar.name)
                assign_op = hvar.assign(dvar)
                sess.run(assign_op)
            print('Done!')

            saver = tf.train.Saver(max_to_keep=3)
            summary_writer = tf.summary.FileWriter(
                logdir=self.log_dir, graph=tf.get_default_graph())
            # the +1 gives an additional smaller batch
            batches_per_epochs = int(
                self.dataset['train']['labels'].shape[0] / self.batch_size) + 1
            print(batches_per_epochs, 'batches (i.e. iterations) per epoch')

            for step in range(self.train_iter_adv):
                i = step % batches_per_epochs
                feed_dict = {self.model.rgb_images: self.dataset['train']['rgb_images'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.depth_images: self.dataset['train']['depth_images'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.labels: self.dataset['train']['labels'][i * self.batch_size:(i + 1) * self.batch_size],
                             self.model.is_training: True}
                # if i == 0:
                if step % 500 == 0:
                    summary, l, acc = sess.run(
                        [self.model.summary_op, self.model.loss, self.model.accuracy], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d]  loss: [%.6f] acc: [%.6f] '
                           % (step, self.train_iter, l, acc))

                if i == 0:
                    # Eval on train
                    print('step %d ' % step)
                    #self.eval_all_double_stream('train', session=sess)
                    self.eval_all_double_stream('test', session=sess)
                    saver.save(
                        sess, os.path.join(self.model_save_path, 'hallucination_eccv'), global_step=step)

                sess.run(self.model.train_op, feed_dict)


if __name__ == '__main__':

    print('Empty')
