""" A simple one-layer classifier for predicting the input tokens from a model's latent representations. """

import tensorflow as tf
from tensorflow.python.ops.init_ops import glorot_uniform_initializer


class LexicalClassifier(object):
    """ A simple, single-layer FFN classifier. """

    def __init__(self, config, vocab_size):
        self.config = config
        self.vocab_size = vocab_size
        self.input_dims = 512
        self.hidden_dims = 512
        self.name = 'lexical_classifier'

        # Build the network
        with tf.variable_scope('lexical_classifier'):
            self.inputs = tf.placeholder(dtype=tf.float32, name='classifier_inputs', shape=[None, None])
            self.labels = tf.placeholder(dtype=tf.int32, name='classifier_labels', shape=[None])
            self.training = tf.placeholder_with_default(False, name='is_training', shape=[])

            self.weights_hidden = tf.get_variable(name='weights_hidden',
                                                  shape=[self.input_dims, self.hidden_dims],
                                                  dtype=tf.float32,
                                                  initializer=glorot_uniform_initializer(),
                                                  trainable=True)

            self.biases_hidden = tf.get_variable(name='biases_hidden',
                                                 shape=[self.hidden_dims],
                                                 dtype=tf.float32,
                                                 initializer=tf.zeros_initializer(),
                                                 trainable=True)

            self.weights_projection = tf.get_variable(name='weights_projection',
                                                      shape=[self.hidden_dims, vocab_size],
                                                      dtype=tf.float32,
                                                      initializer=glorot_uniform_initializer(),
                                                      trainable=True)

            # Instantiate global step variable (used for training)
            self.global_step = tf.get_variable(name='global_step',
                                               shape=[],
                                               dtype=tf.int32,
                                               initializer=tf.zeros_initializer,
                                               trainable=False)

            self.optimizer = self._get_optimizer()

    def _predict(self):
        """ Generates prediction logits at training time. """
        layer1_out = tf.nn.xw_plus_b(self.inputs, self.weights_hidden, self.biases_hidden)
        layer1_out = tf.layers.dropout(layer1_out, rate=0.5, training=self.training)
        layer1_out = tf.nn.relu(layer1_out, name='classifier_hidden')
        logits = tf.matmul(layer1_out, self.weights_projection, name='classifier_logits')
        return logits

    def _get_loss(self):
        """ Returns the training loss incurred by the model to be used in the optimization step. """
        with tf.name_scope('classifier_loss'):
            # Get logits
            logits = self._predict()

            # Calculate loss
            cross_ent_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=
                                                                        tf.one_hot(self.labels, depth=self.vocab_size),
                                                                        logits=logits,
                                                                        dim=-1,
                                                                        name='classifier_cross_ent')

            # Calculate accuracy
            max_logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            correct_predictions = tf.cast(tf.equal(self.labels, max_logits), dtype=tf.int32)
            accuracy = \
                tf.reduce_sum(correct_predictions) / tf.shape(self.labels)[0]  # accuracy per token

        return tf.reduce_mean(cross_ent_loss), accuracy, correct_predictions

    def _get_optimizer(self):
        """ Sets up the model's optimizer. """
        return tf.train.AdamOptimizer(learning_rate=2e-4,
                                      beta1=self.config.adam_beta1,
                                      beta2=self.config.adam_beta2,
                                      epsilon=self.config.adam_epsilon)

    def _optimize(self, optimization_objective):
        """ Optimizes model parameters via stochastic gradient descent. """
        with tf.name_scope('classifier_optimization'):
            # Get trainable variables
            t_vars = tf.trainable_variables()
            # Get gradients
            grads_and_vars = self.optimizer.compute_gradients(optimization_objective)
            grads, _ = zip(*grads_and_vars)
            # Optionally clip gradients to prevent saturation effects
            if self.config.grad_norm_threshold > 0.0:
                grads, _ = tf.clip_by_global_norm(grads, self.config.grad_norm_threshold)
            # Update model parameters by optimizing on obtained gradients
            train_op = self.optimizer.apply_gradients(zip(grads, t_vars), global_step=self.global_step)
        return optimization_objective, train_op

    def train_model(self):
        """ Defines the model graph for the training step. """
        with tf.name_scope('classifier_training'):
            # Optimize
            batch_loss, batch_accuracy, correct_predictions = self._get_loss()
            grads_and_vars, train_op = self._optimize(batch_loss)
        return train_op, batch_loss, batch_accuracy, correct_predictions
