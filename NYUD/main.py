import tensorflow as tf
from model import MultiModal
from solver import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'None', "'train_rgb', 'train_depth', 'train_double_stream','train_double_stream_moddrop', 'train_hallucination', , 'train_hallucination_p2', 'finetune_hallucination',  'test_hallucination', 'train_eccv'")
flags.DEFINE_string('bs', '32', "batch size")
flags.DEFINE_string('lr', '0.0001', "learning rate")
flags.DEFINE_string('it', '2000', "training iterations")
flags.DEFINE_string('noise', '0', "training iterations")
FLAGS = flags.FLAGS


def main(_):
    model = MultiModal(mode=FLAGS.mode, learning_rate=float(FLAGS.lr))
    solver = Solver(model, batch_size=int(FLAGS.bs), train_iter=int(
        FLAGS.it), train_iter_adv=int(FLAGS.it))

    if FLAGS.mode == 'train_rgb':
        solver.train_single_stream(modality='rgb')
    elif 'test_rgb' in FLAGS.mode:  # test also rgb1
        solver.test_single_stream(modality=FLAGS.mode.split('_')[-1])
    elif FLAGS.mode == 'test_ensemble_baseline':  # rgb+rgb1
        solver.test_ensemble_baseline()
    elif FLAGS.mode == 'train_depth':
        solver.train_single_stream(modality='depth')
    elif FLAGS.mode == 'test_depth':
        solver.test_single_stream(modality='depth')
    elif 'train_double_stream' in FLAGS.mode:
        solver.train_double_stream()
    elif 'test_moddrop' in FLAGS.mode:
        solver.test_moddrop(noise=float(FLAGS.noise))
    elif 'train_hallucination' in FLAGS.mode:
        solver.train_hallucination()
    elif FLAGS.mode == 'finetune_hallucination':
        solver.finetune_hallucination()
    elif FLAGS.mode == 'test_hallucination':
        solver.test_hallucination()
    elif FLAGS.mode == 'train_autoencoder':
        solver.train_autoencoder()
    elif FLAGS.mode == 'test_autoencoder':
        solver.test_autoencoder()
    elif FLAGS.mode == 'test_double_stream_with_ae':
        solver.test_double_stream_with_ae()
    elif FLAGS.mode == 'test_double_stream':
        solver.test_double_stream(noise=float(FLAGS.noise))
    elif FLAGS.mode == 'test_disc':
        solver.test_disc(noise=float(FLAGS.noise))
    elif FLAGS.mode == 'train_eccv':
        solver.train_eccv()
    else:
        print 'Unrecognized mode.'


if __name__ == '__main__':
    tf.app.run()
