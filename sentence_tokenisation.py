"""Data must first be converted from raw text files to tokenised files with one sentence per line."""
import nltk
import tensorflow as tf
import glob
import codecs
import re
from multiprocessing import Pool

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('raw_text_dir',
                       'data/webbase_all/',
                       'Path to directory containing the raw text files for processing.')

tf.flags.DEFINE_string('output_tokenised_dir',
                       'data/tokenised_texts',
                       'Path to directory to write files containing tokenised sentences.')


def process_input_file(input_file):
    """Multiprocessing function to tokenise raw file into set of sentences"""

    try:
        print('Processing raw text file {}'.format(input_file))
        with tf.gfile.GFile(input_file, mode="r") as f:
            text = f.read().decode('utf-8')
            text = re.sub('\n', ' ', text).lower()
            text = ''.join(re.findall('[a-z. ]', text))

        sentences = nltk.sent_tokenize(text)
        sentences = [''.join(re.findall('[a-z ]', s)) for s in sentences]
        sentences = list(filter(lambda x: len(x) >= 3, sentences))

        file_name = input_file.split('/')[-1]
        with codecs.open(FLAGS.output_tokenised_dir + '/{}'.format(file_name), "w", "utf-8") as f:
            f.write('\n'.join(sentences))
    except Exception as e:
        print('FAILURE:', process_input_file, e)


if __name__ == '__main__':
    input_files = glob.glob(FLAGS.raw_text_dir + '/*')
    pool = Pool(10)
    pool.map(process_input_file, input_files)
