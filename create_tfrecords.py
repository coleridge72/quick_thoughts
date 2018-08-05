"""Tokenised sentences need to be converted to sharded tf records. This script builds a global vocabulary before writing
n sharded tf-records per tokenised file."""
import tensorflow as tf
import glob
import collections
from multiprocessing import Pool
import numpy as np
import os
import random

FLAGS = tf.flags.FLAGS
SentenceBatch = collections.namedtuple("SentenceBatch", ("ids", "mask"))

tf.flags.DEFINE_string('output_tokenised_dir',
                       'data/tokenised_texts',
                       'Path to directory to read files containing tokenised sentences.')

tf.flags.DEFINE_string('tfrecords_dir',
                       'data/tfrecords/',
                       'Path to directory where tfrecords will be written.')

tf.flags.DEFINE_integer('vocab_size',
                        100000,
                        'Max size of vocab allowed')

tf.flags.DEFINE_integer('max_sentence_length',
                        20,
                        'Max sentence length allowed before constriction')

tf.flags.DEFINE_float('train_proportion',
                      0.8,
                      'Ratio of entire sentences to use for training')

tf.flags.DEFINE_string('output_vocab_dir',
                       'data/',
                       'Directory to save the vocab file')


def write_shards_for_file(args):
    """Writes sharded tf-records for a single tokenised txt file. Run in multi-processing function, with args passed in
    as a tuple.

    Args:
        token_file: Path to the token file.
        vocab: The vocabulary dictionary
        dataset_name: String for tf record identification ie train or dev

    """
    token_file, vocab, dataset_name = args
    processed_sentences = messages_from_file(token_file, vocab)
    file_name = token_file.split('/')[-1]
    num_shards = 5
    borders = np.int32(np.linspace(0, len(processed_sentences), num_shards + 1))
    for i in range(num_shards):
        print('{}: Writing shard {} of {}. for file {} idxs {} - {}'.format(dataset_name, i, num_shards,
                                                                            file_name, borders[i], borders[i + 1]))
        filename = os.path.join(FLAGS.tfrecords_dir, "%s-%.5d-of-%.5d-%s.tfrecord" % (file_name, i,
                                                                                      num_shards, dataset_name))
        shard_points = processed_sentences[borders[i]:borders[i + 1]]
        with tf.python_io.TFRecordWriter(filename) as writer:
            for point in shard_points:
                writer.write(point)


def extract_word_counts(input_file):
    """Multiprocessing function to build and return a dictionary of word counts"""
    print("Counting words in {}".format(input_file))

    try:
        wordcount = collections.Counter()
        for sentence in tf.gfile.FastGFile(input_file):
            wordcount.update(sentence.split())
        return wordcount
    except Exception as e:
        print("FAILED {}", e)
        return 1


def messages_from_file(input_file, vocab):
    """Multiprocessing function given tokenised input file and produces tfrecords"""
    print('processing sentences for {}'.format(input_file))
    processed_sentences = []
    for sentence in tf.gfile.FastGFile(input_file):
        tokens = sentence.split()
        tokens = tokens[:FLAGS.max_sentence_length]
        serialized = create_serialized_example(tokens, vocab)
        processed_sentences.append(serialized)
    return processed_sentences


def create_serialized_example(tokens, vocab):
    """Convert a sentence to a serialised protobuf example"""

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v) for v in value]))

    ids = [vocab.get(w, 0) for w in tokens]
    example = tf.train.Example(features=tf.train.Features(feature={"features": _int64_feature(ids)}))
    return example.SerializeToString()


def print_top_words(counter):
    """Helper function to print the top occurring words in a counter."""
    words = counter.keys()
    freqs = counter.values()
    sorted_indices = np.argsort(freqs)[::-1]
    for w_id, w_index in enumerate(sorted_indices[0:10]):
        print(words[w_index], freqs[w_index])


def parse_single_example(example):
    """Parser function used in data iterator creation"""
    parsed = tf.parse_single_example(example, features={'features': tf.VarLenFeature(tf.int64)})

    features = parsed["features"]
    ids = tf.sparse_tensor_to_dense(features)  # Padding with zeroes.
    mask = tf.sparse_to_dense(features.indices, features.dense_shape,
                              tf.ones_like(features.values, dtype=tf.int32))
    return {'ids': ids, 'mask': mask}


def build_tfrecord_dataset(vocab):
    """Splits the tokenised files into dev and train sets, creates n sharded tf-records for each file. """

    files = glob.glob(FLAGS.output_tokenised_dir + '/*')
    random.shuffle(files)

    num_train = int(FLAGS.train_proportion * len(files))
    train_files = [(f, vocab, 'train') for f in files[:num_train]]
    dev_files = [(f, vocab, 'dev') for f in files[num_train:]]

    print('{} files in Train'.format(len(train_files)))
    print('{} files in Dev'.format(len(dev_files)))

    pool = Pool(10)
    pool.map(write_shards_for_file, train_files)
    pool.map(write_shards_for_file, dev_files)


def build_vocabulary():
    """Build a vocabulary across input files."""

    files = glob.glob(FLAGS.output_tokenised_dir + '/*')

    pool = Pool(10)
    word_counts = pool.map(extract_word_counts, files)

    summed_wordcounts = collections.Counter()
    for i, counts in enumerate(word_counts):
        print('Summing dictionary {} of {}'.format(i, len(word_counts)))
        summed_wordcounts += counts

    print("\nTotal Wordcounts")
    print_top_words(summed_wordcounts)

    words = summed_wordcounts.keys()
    freqs = summed_wordcounts.values()
    sorted_indices = np.argsort(freqs)[::-1]

    # Create a vocabulary from the word counts
    vocab = collections.OrderedDict()
    vocab['<unk>'] = 0
    for w_id, w_index in enumerate(sorted_indices[0:FLAGS.vocab_size - 1]):
        vocab[words[w_index]] = w_id + 1  # 0: <unk>
    print('Created Vocab of size {}'.format(len(vocab)))

    # Write the vocabulary to output directory.
    vocab_file = os.path.join(FLAGS.output_vocab_dir, "vocab_{}.txt".format(len(vocab)))
    with tf.gfile.FastGFile(vocab_file, "w") as f:
        f.write("\n".join(vocab.keys()))
    print("Vocab saved in file {}".format(vocab_file))

    # Write all the words by frequency.
    word_counts_file = os.path.join(FLAGS.output_vocab_dir, "word_counts.txt")
    with tf.gfile.FastGFile(word_counts_file, "w") as f:
        for i in sorted_indices:
            f.write("%s %d\n" % (words[i], freqs[i]))
    print("Wrote word counts file to {}".format(word_counts_file))

    return vocab


if __name__ == '__main__':
    vocab = build_vocabulary()
    build_tfrecord_dataset(vocab)
