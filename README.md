# Quick Thoughts

Based on work by Lajanugen Logeswaran & Honglak Lee in "An Efficient Framework for Learning Sentence Representations" published at ICLR 2018.

This implementation is based on the original, but swaps out the RNN encoder for a Transformer. I also use only a single encoder, rather than one for the query and one for the context. (In other words encoders f and g share parameters in Figure 1. See https://arxiv.org/pdf/1803.02893.pdf)

## How to use
1. Raw text -> Sentence per line: `sentence_tokenisation.py` converts raw text files to text files with a single sentence per line.
2. Sentences -> TFRecords: `create_tfrecords.py` first builds a vocab over all sentences before serialising them as integer (index) lists into tfrecords.
3. Train: `train.py` runs the model over the training set. The tf-records are not shuffled and the loss computed by the model predicting which sentences in a batch immediately precede/follow every other sentence.

## Analysis
`analysis.py` contains code to explore the patterns learnt by a trained model. A set of sentences are transformed to vectors by the model. By entering a custom sentence, you can search through the text by printing sentences that have the largest dot product between the respective encodings.
