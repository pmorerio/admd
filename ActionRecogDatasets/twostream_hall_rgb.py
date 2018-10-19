# twostream model - RGB and hall

# import ipdb
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from codebase import utils
from codebase import parsers
from codebase import restorers

# tensorflow models_dir ##################################################
import sys
sys.path.insert(0, utils.tensorflow_models_dir)
import nets.resnet_v1_two_stream as resnet_v1
############################################################################

hall_ckpt = './checkpoint/ntu/s2-gan-hall-depth_01012018_010101__dset_ntu_eval_mode_cross_view/test/model.ckpt'
rgb_ckpt = './checkpoint/ntu/s1_train_rgb_01012018_010101__dset_ntu_eval_mode_cross_view/test/model.ckpt'


def train(exp_id, files, args):
    log_path = './log'
    ckpt_path = './checkpoint'

    # dataset ######################################################
    train_filenames, val_filenames, test_filenames = utils.get_tfrecords(
        args.eval_mode, files['data'], dataset=args.dset)
    n_classes = utils.get_n_classes(args.dset)

    with tf.device('/cpu:0'):
        dset_train = tf.contrib.data.TFRecordDataset(
            train_filenames, compression_type="GZIP")
        dset_train = dset_train.map(
            lambda x: parsers._parse_fun_one_mod(x, is_training=True, modality='rgb'))
        seed = tf.placeholder(tf.int64, shape=())
        dset_train = dset_train.shuffle(100, seed=seed)
        dset_train = dset_train.batch(args.batch_sz)

        if val_filenames:
            dset_val = tf.contrib.data.TFRecordDataset(
                val_filenames, compression_type="GZIP")
            dset_val = dset_val.map(
                lambda x: parsers._parse_fun_one_mod(x, is_training=False, modality='rgb'))
            dset_val = dset_val.batch(args.batch_sz)

        dset_test = tf.contrib.data.TFRecordDataset(
            test_filenames, compression_type="GZIP")
        dset_test = dset_test.map(
            lambda x: parsers._parse_fun_one_mod(x, is_training=False, modality='rgb'))
        dset_test = dset_test.batch(args.batch_sz)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.contrib.data.Iterator.from_string_handle(handle,
                                                               dset_train.output_types, dset_train.output_shapes)

        train_iterator = dset_train.make_initializable_iterator()
        if val_filenames:
            val_iterator = dset_val.make_initializable_iterator()
        test_iterator = dset_test.make_initializable_iterator()

        next_element = iterator.get_next()
        images_rgb_stacked = next_element[0]

        if args.dset == 'uwa3dii':  # because tfrecords labels are [1,30]
            labels = next_element[1] - 1
        elif 'ntu' in args.dset or args.dset == 'nwucla':
            labels = next_element[1]
        labels = tf.reshape(labels, [-1])
        labels = tf.one_hot(labels, n_classes)

        rgb_stack_shape = tf.shape(images_rgb_stacked)
        # reshape to [batch * pooled_frames, h,w,c]
        images_rgb = tf.reshape(images_rgb_stacked, [
            rgb_stack_shape[0] * rgb_stack_shape[1], 224, 224, 3])

    # -----TF.CONFIGPROTO------###########################################
    tf_config = tf.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    # tf Graph input ##############################################
    with tf.device(args.gpu0):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            is_training = tf.placeholder(tf.bool, [])
            nr_frames = parsers.time_bottleneck

            net_hall_out, endpoints_hall = resnet_v1.resnet_one_stream_main(
                images_rgb, nr_frames, num_classes=n_classes, scope='resnet_v1_50_hall', gpu_id=args.gpu0, is_training=is_training, bottleneck=True)
            net_rgb_out, endpoints_rgb = resnet_v1.resnet_one_stream_main(
                images_rgb, nr_frames, num_classes=n_classes, scope='resnet_v1_50_rgb', gpu_id='/gpu:1', is_training=is_training, bottleneck=False)

            # predictions for each video are the avg of frames' predictions
            # TRAIN ###############################
            net_hall_train = tf.reshape(
                net_hall_out, [-1, parsers.time_bottleneck, n_classes])
            net_hall_train = tf.reduce_mean(net_hall_train, axis=1)
            net_rgb_train = tf.reshape(
                net_rgb_out, [-1, parsers.time_bottleneck, n_classes])
            net_rgb_train = tf.reduce_mean(net_rgb_train, axis=1)
            net_combined_train = tf.add(net_hall_train, net_rgb_train) / 2.0

            # TEST ###############################
            net_rgb_test = tf.reshape(
                net_rgb_out, [-1, parsers.time_bottleneck, n_classes])
            net_rgb_test = tf.reduce_mean(net_rgb_test, axis=1)
            net_hall_test = tf.reshape(
                net_hall_out, [-1, parsers.time_bottleneck, n_classes])
            net_hall_test = tf.reduce_mean(net_hall_test, axis=1)
            net_combined_test = tf.add(net_rgb_test, net_hall_test) / 2.0

            # losses ##########################################################
            loss = slim.losses.softmax_cross_entropy(
                net_combined_train, labels)
            loss_hall = slim.losses.softmax_cross_entropy(
                net_hall_train, labels)
            loss_rgb = slim.losses.softmax_cross_entropy(
                net_rgb_train, labels)

            optimizer = tf.train.AdamOptimizer(
                learning_rate=args.learning_rate)
            minimizing = slim.learning.create_train_op(loss, optimizer)

            acc_hall_train = utils.accuracy(net_hall_train, labels)
            acc_rgb_train = utils.accuracy(net_rgb_train, labels)
            acc_combined_train = utils.accuracy(net_combined_train, labels)

            n_correct_hall = tf.reduce_sum(
                tf.cast(utils.correct_pred(net_hall_test, labels), tf.float32))
            n_correct_rgb = tf.reduce_sum(
                tf.cast(utils.correct_pred(net_rgb_test, labels), tf.float32))
            n_correct_combined = tf.reduce_sum(
                tf.cast(utils.correct_pred(net_combined_test, labels), tf.float32))

    summ_loss_combined = tf.summary.scalar('loss', loss)
    summ_loss_hall = tf.summary.scalar('loss_hall', loss_hall)
    summ_loss_rgb = tf.summary.scalar('loss_rgb', loss_rgb)
    summ_acc_train_rgb = tf.summary.scalar('acc_train_rgb', acc_rgb_train)
    summ_acc_train_hall = tf.summary.scalar(
        'acc_train_hall', acc_hall_train)
    summ_acc_train_combined = tf.summary.scalar(
        'acc_train_combined', acc_combined_train)
    summ_train = tf.summary.merge(
        [summ_acc_train_rgb, summ_acc_train_hall, summ_acc_train_combined, summ_loss_hall, summ_loss_rgb, summ_loss_combined])

    accuracy_value_ = tf.placeholder(tf.float32, shape=())
    summ_acc_val_rgb = tf.summary.scalar('acc_val_rgb', accuracy_value_)
    summ_acc_val_hall = tf.summary.scalar('acc_val_hall', accuracy_value_)
    summ_acc_val_combined = tf.summary.scalar(
        'acc_val_combined', accuracy_value_)
    summ_acc_test_rgb = tf.summary.scalar('acc_test_rgb', accuracy_value_)
    summ_acc_test_hall = tf.summary.scalar(
        'acc_test_hall', accuracy_value_)
    summ_acc_test_combined = tf.summary.scalar(
        'acc_test_combined', accuracy_value_)
    test_saver = tf.train.Saver(max_to_keep=3)

    with tf.Session(config=tf_config) as sess:
        train_handle = sess.run(train_iterator.string_handle())
        if val_filenames:
            val_handle = sess.run(val_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        summary_writer = tf.summary.FileWriter(
            os.path.join(log_path, args.dset, exp_id), sess.graph)

        f_log = open(os.path.join(
            log_path, args.dset, exp_id, 'log.txt'), 'a')
        utils.double_log(f_log, '\n###############################################\n' +
                         exp_id + '\n#####################################\n')
        f_log.write(' '.join(sys.argv[:]) + '\n')
        f_log.flush()

        sess.run(tf.global_variables_initializer())
        restorers.restore_weights_hall_rgb(sess, rgb_ckpt, hall_ckpt)

        def val_test(value_step, mode='val'):
            if mode == 'val':
                if not val_filenames:
                    return -1
                utils.double_log(f_log, "eval validation set \n")
                sess.run(val_iterator.initializer)
                step_handle = val_handle
                step_samples = len(val_filenames)
                step_summ_rgb = summ_acc_val_rgb
                step_summ_hall = summ_acc_val_hall
                step_summ_combined = summ_acc_val_combined
            elif mode == 'test':
                utils.double_log(f_log, "eval test set \n")
                sess.run(test_iterator.initializer)
                step_handle = test_handle
                step_samples = len(test_filenames)
                step_summ_rgb = summ_acc_test_rgb
                step_summ_hall = summ_acc_test_hall
                step_summ_combined = summ_acc_test_combined
            try:
                accum_correct_rgb = accum_correct_hall = accum_correct_combined_val = 0
                while True:
                    n_correct_rgb1, n_correct_hall1, n_correct_combined1 = sess.run(
                        [n_correct_rgb, n_correct_hall, n_correct_combined], feed_dict={handle: step_handle, is_training: False})
                    accum_correct_rgb += n_correct_rgb1
                    accum_correct_hall += n_correct_hall1
                    accum_correct_combined_val += n_correct_combined1
            except tf.errors.OutOfRangeError:
                acc_rgb = accum_correct_rgb / step_samples
                acc_hall = accum_correct_hall / step_samples
                acc_combined = accum_correct_combined_val / step_samples
                sum_rgb_acc = sess.run(step_summ_rgb, feed_dict={
                    accuracy_value_: acc_rgb})
                summary_writer.add_summary(sum_rgb_acc, value_step)
                sum_hall_acc = sess.run(step_summ_hall, feed_dict={
                    accuracy_value_: acc_hall})
                summary_writer.add_summary(sum_hall_acc, value_step)
                sum_combined_acc = sess.run(step_summ_combined, feed_dict={
                    accuracy_value_: acc_combined})
                summary_writer.add_summary(sum_combined_acc, value_step)
                utils.double_log(
                    f_log, 'Hall accuracy = %s \n' % str(acc_hall))
                utils.double_log(
                    f_log, 'RGB accuracy = %s \n' % str(acc_rgb))
                utils.double_log(f_log, 'combined accuracy = %s \n' %
                                 str(acc_combined))
                return acc_combined

        if args.just_eval:
            val_test(-1, mode='test')
            f_log.close()
            summary_writer.close()
            return

        val_test(-1, mode='val')
        val_test(-1, mode='test')
        n_step = 0
        best_acc = best_epoch = best_step = -1
        for epoch in range(args.n_epochs):
            utils.double_log(f_log, 'epoch %s \n' % str(epoch))
            sess.run(train_iterator.initializer, feed_dict={seed: epoch})
            try:
                while True:
                    print(n_step)
                    if n_step % 100 == 0:  # get summaries
                        _, summary = sess.run(
                            [minimizing, summ_train], feed_dict={handle: train_handle, is_training: True})
                        summary_writer.add_summary(summary, n_step)
                    else:
                        sess.run([minimizing], feed_dict={
                            handle: train_handle, is_training: True})
                    n_step += 1
            except tf.errors.OutOfRangeError:
                acc_validation = val_test(n_step, mode='val')

            if val_filenames:
                acc_epoch = acc_validation
            else:
                continue
            if acc_epoch >= best_acc:
                best_acc = acc_epoch
                best_epoch = epoch
                best_step = n_step
                test_saver.save(
                    sess, os.path.join(ckpt_path, args.dset, exp_id, 'test/model.ckpt'), global_step=n_step)

        utils.double_log(f_log, "Optimization Finished!\n")
        if val_filenames:  # restore best validation model
            utils.double_log(f_log, str(
                "Best Validation Accuracy: %f at epoch %d \n" % (best_acc, best_epoch)))
            variables_to_restore = slim.get_variables_to_restore()
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                ckpt_path, args.dset, exp_id, 'test/model.ckpt-' + str(best_step)))
        else:
            test_saver.save(
                sess, os.path.join(ckpt_path, args.dset, exp_id, 'test/model.ckpt'), global_step=n_step)

        val_test(n_step + 1, mode='test')
        f_log.close()
        summary_writer.close()


def main():
    args = utils.get_arguments()
    experiment_id_prefix = 'twostream_hall_rgb'
    exp_id = utils.create_folders(prefix=experiment_id_prefix,
                                  dset=args.dset, eval_mode=args.eval_mode)
    files = utils.load_files_paths(args.dset)
    train(exp_id, files, args)


if __name__ == '__main__':
    main()
