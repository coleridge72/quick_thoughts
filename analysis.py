"""Rough script to experiment with a trained model."""
from model import QuickThoughtsModel
import tensorflow as tf
from create_tfrecords import parse_single_example
import glob
import numpy as np
import re
from tensorflow.contrib.data import padded_batch_and_drop_remainder

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 300, 'Batch Size')
tf.flags.DEFINE_integer('embedding_size', 100, 'Size of sentence embedding')
tf.flags.DEFINE_integer("context_size", 1, "Prediction context size")
tf.flags.DEFINE_integer('wembedding_size', 100, 'Word embeddings dimension size')
tf.flags.DEFINE_string('tfrecords_input', 'data/tfrecords/', 'Path to directory where tfrecords will be read from.')
tf.flags.DEFINE_string('logdir', 'logdir', 'Path to logdir.')
tf.flags.DEFINE_string('vocab_file', 'data/vocab_100000.txt', 'Path to vocab file')
tf.flags.DEFINE_string('checkpoint_to_load', 'YOUR CHECKPOINT HERE', 'Path to checkpoint file')

PADDED_SHAPES = ({'ids': [FLAGS.max_sentence_length], 'mask': [FLAGS.max_sentence_length]})

vocab = []
for i, word in enumerate(tf.gfile.FastGFile(FLAGS.vocab_file)):
    vocab.append(word.decode("utf-8").strip())


def num_tfrecords(files):
    """Count tfrecords in data"""
    return sum(sum((1 for _ in tf.python_io.tf_record_iterator(str(file_)))) for file_ in files)


def ids_to_sentence(ids):
    """Convert list of vocab indexes into sentence"""
    return ' '.join([vocab[i] for i in ids])


def sentence_to_ids(sentence, vocab):
    """Convert a sentence string to a list of ints"""
    sentence = sentence.lower()
    sentence = ''.join(re.findall('[a-z ]', sentence))

    def word_to_idx(word):
        if word not in vocab: return 0
        else: return vocab.index(word)

    ids = [word_to_idx(w) for w in sentence.split()[:FLAGS.max_sentence_length]]
    mask_len = len(ids)
    while len(ids) < FLAGS.max_sentence_length:
        ids.append(0)
    mask = [int(i < mask_len) for i in range(FLAGS.max_sentence_length)]
    return ids, mask


def construct_encoding_dictionary():
    """Build a dictionary of sentence:encoding pairs"""
    sentence_encoding = {}
    dev_graph = tf.Graph()
    with dev_graph.as_default():
        train_files = glob.glob('data/tfrecords/*train.tfrecord')
        train_data = tf.data.TFRecordDataset(train_files) \
            .map(parse_single_example) \
            .apply(padded_batch_and_drop_remainder(FLAGS.batch_size, PADDED_SHAPES))
        training_iterator = train_data.make_initializable_iterator()

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

        dev_saver = tf.train.Saver(var_list=tf.global_variables())

        devSess = tf.Session()
        dev_saver.restore(devSess, FLAGS.checkpoint_to_load)
        devSess.run(training_iterator.initializer)
        i =0
        while True:
            i+=1
            print('Collecting {}'.format(i))
            try:
                ids_, mask_, loss_, encoding_ = devSess.run([sentences['ids'], sentences['mask'], loss, encoding])
            except tf.errors.OutOfRangeError:
                break
            sents = [(ids_to_sentence(x[:sum(y)]), z) for (x, y, z) in zip(ids_, mask_, encoding_)]
            sentence_encoding.update(sents)

    return sentence_encoding


def custom_search(sentence_encoding, vocab):
    """Encodes a custom sentence and searches against other sentences for closest match."""
    dev_graph = tf.Graph()
    with dev_graph.as_default():
        train_files = glob.glob('data/tfrecords/*train.tfrecord')
        train_data = tf.data.TFRecordDataset(train_files) \
            .map(parse_single_example) \
            .apply(padded_batch_and_drop_remainder(FLAGS.batch_size, PADDED_SHAPES))
        training_iterator = train_data.make_initializable_iterator()

        global_step = tf.Variable(0, trainable=False)
        model = QuickThoughtsModel()
        ids_ph = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size ,FLAGS.max_sentence_length], name='ids')
        mask_ph = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.max_sentence_length], name='ids_mask')
        encoding = model.encode(ids_ph, mask_ph)
        loss = model.loss(encoding)
        opt = tf.train.AdamOptimizer(0.001)
        gradients = opt.compute_gradients(loss)
        metric_summary = tf.summary.merge_all()

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            step = opt.apply_gradients(gradients, global_step=global_step)

        dev_saver = tf.train.Saver(var_list=tf.global_variables())

        devSess = tf.Session()
        dev_saver.restore(devSess, FLAGS.checkpoint_to_load)
        devSess.run(training_iterator.initializer)

        while True:
            try:
                sentence = input('\nEnter custom text to search:')
            except Exception as e:
                print('Failed, try again "{}"'.format(e))
            ids, mask = sentence_to_ids(sentence, vocab)
            ids = [ids] * FLAGS.batch_size
            mask = [mask] * FLAGS.batch_size
            encoding_ = devSess.run(encoding, {ids_ph: ids, mask_ph: mask})
            print("Similar sentences to: '{}'".format(sentence))
            print_closest_matches_to_sentence(encoding_, sentence_encoding)


def print_closest_matches_to_sentence(encoding, sentence_encodings):
    """Print the closest matches to the given sentence encoding"""
    keys = sentence_encodings.keys()
    mat = np.array(sentence_encodings.values())
    mat_sim = np.matmul(encoding, mat.T)[0]

    similar_ids = np.argsort(mat_sim)[::-1]
    for id in similar_ids[:5]:
        print("\t{}: {}".format(mat_sim[id], keys[id]))


def print_similar_sentences(sentence_encoding):
    """Given the sentence:encoding dict, print out the most similar sentences from the dict."""
    keys = sentence_encoding.keys()
    mat = np.array(sentence_encoding.values())
    mat_sim = np.matmul(mat, mat.T)

    for row in range(10):
        print('\n\nOriginal sentence: {}'.format(keys[row]))
        similar_ids = np.argsort(mat_sim[row])[::-1]
        for id in similar_ids[:5]:
            print("\t{}: {}".format(mat_sim[row][id], keys[id]))


if __name__ == '__main__':
    sentence_encoding = construct_encoding_dictionary()
    print_similar_sentences(sentence_encoding)
    custom_search(sentence_encoding, vocab)
