"""Main Quick thoughts model definition, uses transformer network with shared weights."""
from tensor2tensor.models import transformer
import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS


class QuickThoughtsModel(object):
    """Encodes sentences into distributed vectors, predicts context sentences from batch of candidates."""

    def encode(self, indexed_batch, mask):
        """Take in a batch of encoded sentences formed of word indices, return a batch of sentence vectors."""

        word_embeddings = tf.get_variable(
                name='word_embeddings',
                shape=[FLAGS.vocab_size, FLAGS.wembedding_size],
                initializer=tf.random_uniform_initializer(),
                trainable=True)

        # Mask the padded word embeddings
        words = tf.nn.embedding_lookup(word_embeddings, indexed_batch)
        words = tf.multiply(words, tf.cast(tf.expand_dims(mask, -1), dtype=tf.float32))
        words = tf.expand_dims(words, 1)

        transformer_params = transformer.transformer_big()
        transformer_params.num_heads = 5
        transformer_params.hidden_size = FLAGS.wembedding_size

        # Transformer encoder outputs shape [BatchSize MaxLength HiddenSize]
        tfmr = transformer.Transformer(transformer_params, mode=tf.estimator.ModeKeys.TRAIN)
        target_space_id = tf.constant(1, dtype=tf.int32)
        encoder_output, _ = tfmr.encode(words, target_space_id, transformer_params)

        # Use a linear transform to map onto shape [BatchSize, SentenceEmbeddingSize]
        encoder_output = tf.reshape(encoder_output, [FLAGS.batch_size, -1])
        matrix_shape = [FLAGS.wembedding_size * FLAGS.max_sentence_length, FLAGS.embedding_size]
        matrix = tf.random_normal(matrix_shape, dtype=tf.float32, name='linear_layer')
        linear_transform = tf.Variable(matrix)
        sentence_embeddings = tf.matmul(encoder_output, linear_transform)
        return sentence_embeddings

    def loss(self, encoded_batch):
        """Compute the batch-wise loss on encoded sentences"""

        # Compute a matrix of sentence dot products.
        scores = tf.matmul(encoded_batch, encoded_batch, transpose_b=True)
        scores = tf.matrix_set_diag(scores, np.zeros(FLAGS.batch_size))

        # Define the targets as sentences before and after each sentence (the context)
        targets_np = np.zeros((FLAGS.batch_size, FLAGS.batch_size))
        ctxt_sent_pos = range(-FLAGS.context_size, FLAGS.context_size + 1)
        ctxt_sent_pos.remove(0)
        for ctxt_pos in ctxt_sent_pos:
            targets_np += np.eye(FLAGS.batch_size, k=ctxt_pos)

        # Normalise the matrices by row
        targets_np_sum = np.sum(targets_np, axis=1, keepdims=True)
        targets_np = targets_np / targets_np_sum
        targets = tf.constant(targets_np, dtype=tf.float32)

        # Compute cross entropy between the scores and targets
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=scores)
        loss = tf.reduce_mean(losses)
        tf.summary.scalar("x-entropy", loss)
        return loss
