#!/usr/bin/env python
"""emoji2vec model implemented in TensorFlow.

File also contains a ModelParams class, which is a convenience wrapper for all the parameters to the model.
Details of the model can be found below.
"""

# External dependencies
import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics
import gensim.models as gs

from trainer import Trainer
from batcher import BatchNegSampler


class Emoji2Vec:
    """Class for representing the model in TensorFlow."""
    # TODO(beisner): Describe the model in more detail here

    # define the model
    def __init__(self, hp, num_emoji, embeddings_array, use_embeddings=True):
        """Constructor for the Emoji2Vec model

        Args:
            hp: Parameters for the model
            num_emoji: Number of emoji we will ultimately train
            embeddings_array: For quick training, we inject a constant array into TensorFlow graph consisting
                of vector sums of the embeddings
            use_embeddings: If True, embeddings must be passed in, but the model will not accept arbitrary queries
                If false, it will accept arbitrary queries
        """

        self.hp = hp
        self.num_cols = num_emoji
        self.embeddings_array = embeddings_array


        # Emoji indices in current batch
        self.col = tf.placeholder(tf.int32, shape=[None], name='col')

        # Correlation between an emoji and a phrase
        self.y = tf.placeholder(tf.float32, shape=[None], name='y')

        # Column embeddings (here emoji representations)
        self.V = tf.Variable(tf.random_uniform([num_emoji, hp["out_dim"]], -0.1, 0.1), name="V")

        # original phrase embeddings from Word2Vec, dependent on parameter
        if use_embeddings:
            # Phrase indices in current batch
            self.row = tf.placeholder(tf.int32, shape=[None], name='row')
            # constant embeddings
            tf_embeddings = tf.constant(embeddings_array)
            orig_vec = tf.nn.embedding_lookup(tf_embeddings, self.row)
        else:
            self.orig_vec = tf.placeholder(tf.float32, shape=[None, hp["in_dim"]], name='orig_vec')
            orig_vec = self.orig_vec

        v_row = orig_vec

        v_col = tf.nn.embedding_lookup(self.V, self.col)
        v_col = tf.nn.dropout(v_col, (1 - hp["dropout"]))

        # Calculate the predicted score, a.k.a. dot product (here)
        self.score = tf.reduce_sum(tf.multiply(v_row, v_col), 1)

        # Probability of match
        self.prob = self.score / (tf.norm(v_row) * tf.norm(v_col))
        # Calculate the cross-entropy loss
        #self.loss = self.y * tf.reduce_sum(tf.multiply(v_row, v_col), 1)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score, labels=self.y)

    # train the model using the appropriate parameters
    def train(self, corpus, session, datasets):
        """Train the model on a given knowledge base

        Args:
            kb: Knowledge Base, will only train on training examples
            hooks: Hooks to print out metrics at various intervals
            session: TensorFlow session

        """
        # Train the model
        optimizer = tf.train.AdamOptimizer(self.hp["learning_rate"])

        trainer = Trainer(optimizer, self.hp["max_epochs"])
        trainer(corpus, placeholders=[self.col, self.row, self.y], loss=self.loss, model=self.score,
                session=session)

    def predict(self, session, dset, threshold=None):
        """Generate predictions on a given set of examples using TensorFlow

        Args:
            session: TensorFlow session
            dset: Dataset tuple (emoji_ix, phrase_ix, truth)
            threshold: Threshold for classification

        Returns:
            Returns predicted values for an example, as well as the true value
        """
        phr_ix, em_ix, truth = dset

        y_pred = session.run(self.score, feed_dict={
            self.col: em_ix,
            self.row: phr_ix
        })
        y_pred = y_pred / np.linalg.norm(y_pred)

        if threshold:
        	y_pred = [1 if y > threshold else 0 for y in y_pred]
        y_true = np.asarray(truth).astype(int)
        return y_pred, y_true

    def accuracy(self, session, dset, threshold=0.5):
        """Calculate the accuracy of a dataset at a given threshold.

        Args:
            session: TensorFlow session
            dset: Dataset tuple (emoji_ix, phrase_ix, truth)
            threshold: Threshold for classification

        Returns:
            Accuracy
        """
        y_pred, y_true = self.predict(session, dset, threshold)
        return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    def f1_score(self, session, dset, threshold=0.5):
        """Calculate the f1 score of a dataset at a given classification threshold.

        Args:
            session: TensorFlow session
            dset: Dataset tuple (emoji_ix, phrase_ix, truth)
            threshold: Threshold for classification

        Returns:
            F1 score
        """
        y_pred, y_true = self.predict(session, dset, threshold)
        return metrics.f1_score(y_true=y_true, y_pred=y_pred)

    def auc(self, session, dset):
        """Calculates the Area under the Curve for the f1 at various thresholds

        Args:
            session: TensorFlow session
            dset: Dataset tuple (emoji_ix, phrase_ix, truth)

        Returns:

        """
        y_pred, y_true = self.predict(session, dset)
        return metrics.roc_auc_score(y_true, y_pred)

    def roc_vals(self, session, dset):
        """Generates a receiver operating curve for the dataset

        Args:
            session: TensorFlow session
            dset: Dataset tuple (emoji_ix, phrase_ix, truth)

        Returns:
            Points on the curve
        """
        y_pred, y_true = self.predict(session, dset)
        return metrics.roc_curve(y_true, y_pred)

    def create_gensim_files(self, sess, model_folder, ind2emoj, out_dim):
        """Given a trained session and a destination path (model_folder), generate the gensim binaries
        for a model.

        Args:
            sess: A trained TensorFlow session
            model_folder: Folder in which to generate the files
            ind2emoj: Mapping from indices to emoji
            out_dim: Output dimension of the emoji vectors

        Returns:

        """
        vecs = sess.run(self.V)
        txt_path = model_folder + '/emoji2vec.txt'
        bin_path = model_folder + '/emoji2vec.bin'
        f = open(txt_path, 'w')
        f.write('%d %d\n' % (len(vecs), out_dim))
        for i in range(len(vecs)):
            f.write(ind2emoj[i] + ' ')
            for j in range(out_dim):
                f.write(str.format('{} ', vecs[i][j]))
            f.write('\n')
        f.close()

        e2v = gs.KeyedVectors.load_word2vec_format(txt_path, binary=False)
        e2v.save_word2vec_format(bin_path, binary=True)

        return e2v
