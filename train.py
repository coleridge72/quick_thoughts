"""Main training script"""
from model import QuickThoughtsModel
import tensorflow as tf
from create_tfrecords import parse_single_example
import glob
import numpy as np
from tensorflow.contrib.data import padded_batch_and_drop_remainder

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 300, 'Batch Size')
tf.flags.DEFINE_integer('embedding_size', 100, 'Size of sentence embedding')
tf.flags.DEFINE_integer("context_size", 1, "Prediction context size")
tf.flags.DEFINE_integer('wembedding_size', 100, 'Word embeddings dimension size')
tf.flags.DEFINE_string('tfrecords_input', 'data/tfrecords/', 'Path to directory where tfrecords will be read from.')
tf.flags.DEFINE_string('logdir', 'logdir', 'Path to logdir.')
tf.flags.DEFINE_string('vocab_file', 'data/vocab_100000.txt', 'Path to vocab file')

PADDED_SHAPES = ({'ids': [FLAGS.max_sentence_length], 'mask': [FLAGS.max_sentence_length]})
summary_writer_train = tf.summary.FileWriter('{}/train'.format(FLAGS.logdir))
summary_writer_dev = tf.summary.FileWriter('{}/dev'.format(FLAGS.logdir))

vocab = []
for i, word in enumerate(tf.gfile.FastGFile(FLAGS.vocab_file)):
    vocab.append(word.decode("utf-8").strip())


def num_tfrecords(files):
    return sum(sum((1 for _ in tf.python_io.tf_record_iterator(str(file_)))) for file_ in files)


train_graph = tf.Graph()
with train_graph.as_default():
    train_files = glob.glob('data/tfrecords/*train.tfrecord')
    num_train = num_tfrecords(train_files)
    print('Train Examples: {}'.format(num_train))
    train_data = tf.data.TFRecordDataset(train_files) \
        .map(parse_single_example) \
        .apply(padded_batch_and_drop_remainder(FLAGS.batch_size, PADDED_SHAPES))
    training_iterator = train_data.make_initializable_iterator()

    dev_files = glob.glob('data/tfrecords/*dev.tfrecord')
    num_dev = num_tfrecords(dev_files)
    print('Dev Examples: {}'.format(num_dev))
    dev_data = tf.data.TFRecordDataset(dev_files) \
        .map(parse_single_example) \
        .apply(padded_batch_and_drop_remainder(FLAGS.batch_size, PADDED_SHAPES))
    dev_iterator = dev_data.make_initializable_iterator()

    global_step = tf.Variable(0, trainable=False)
    model = QuickThoughtsModel()
    sentences = training_iterator.get_next()
    encoding = model.encode(sentences['ids'], sentences['mask'])
    loss = model.loss(encoding)
    opt = tf.train.AdamOptimizer(0.001)
    gradients = opt.compute_gradients(loss)
    metric_summary = tf.summary.merge_all()

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        step = opt.apply_gradients(gradients, global_step=global_step)

    train_saver = tf.train.Saver(var_list=tf.global_variables())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    while True:
        # Full step over train set
        print('New Training Epoch')
        sess.run(training_iterator.initializer)
        train_iters = 0
        train_losses = []
        while True:
            try:
                _loss, _step, _global_step = sess.run([loss, step, global_step])
                print("Step {}. {}/{}. Loss {}".format(_global_step, train_iters, num_train,  _loss))
                train_iters += FLAGS.batch_size
                train_losses.append(_loss)
                if len(train_losses) == 100:
                    loss_sum = tf.Summary()
                    loss_sum.value.add(tag='x-entropy', simple_value=np.mean(train_losses))
                    summary_writer_train.add_summary(loss_sum, _global_step)
                    train_losses = []
                if _global_step == 1000:
                    checkpoint = train_saver.save(sess, FLAGS.logdir + '/save/checkpoint', global_step=global_step)
            except tf.errors.OutOfRangeError:
                break

        checkpoint = train_saver.save(sess, FLAGS.logdir + '/save/checkpoint', global_step=global_step)
