import os
import numpy as np
import tensorflow as tf
from base_model import Model
from params_init import random_uniform_initializer, random_normal_initializer, xavier_initializer, \
    random_truncated_normal_initializer
from utils.general_utils import Progbar
from utils.general_utils import get_minibatches, print_confusion_matrix, write_wrong_predictions, get_weighted_f1_score
from utils.feature_extraction import DataConfig


class TextCNN(Model):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """


    # TODO change function invocations to include last 2 arguments
    def __init__(self, config, idx2label, word_embeddings, label2idx, char_embeddings):
        self.word_embeddings = word_embeddings
        self.char_embeddings = char_embeddings
        self.idx2label = idx2label
        self.class_names = map(lambda x: x[0], sorted(label2idx.items(), key=lambda x: x[1]))
        self.config = config
        self.build()


    def add_placeholders(self):

        with tf.variable_scope("input_placeholders"):
            self.word_input_placeholder = tf.placeholder(shape=[None, self.config.max_seq_len],
                                                         dtype=tf.int32, name="batch_word_indices")
        with tf.variable_scope("input_char_placeholders"):
            self.char_input_placeholder = tf.placeholder(
                shape=[None, self.config.max_seq_len, self.config.max_word_len],
                dtype=tf.int32, name="batch_char_indices")
        with tf.variable_scope("label_placeholders"):
            self.labels_placeholder = tf.placeholder(shape=[None, self.config.num_classes],
                                                     dtype=tf.float32, name="batch_one_hot_targets")
        with tf.variable_scope("word_regularization"):
            self.dropout_placeholder_word = tf.placeholder(shape=(), dtype=tf.float32, name="dropout")

        with tf.variable_scope("char_regularization"):
            self.dropout_placeholder_char = tf.placeholder(shape=(), dtype=tf.float32, name="dropout")

        with tf.variable_scope("fc_regularization"):
            self.dropout_placeholder_fc = tf.placeholder(shape=(), dtype=tf.float32, name="dropout")

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.class_weights = tf.constant([self.config.class_weights], dtype=tf.float32)


    def create_feed_dict(self, inputs_batch, labels_batch=None, keep_prob_word=1, keep_prob_char=1, keep_prob_fc=1,
                         is_training=False):

        feed_dict = {
            self.word_input_placeholder: inputs_batch[0],
            self.char_input_placeholder: inputs_batch[1],
            self.dropout_placeholder_word: keep_prob_word,
            self.dropout_placeholder_char: keep_prob_char,
            self.dropout_placeholder_fc: keep_prob_fc,
            self.is_training: is_training
        }

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict


    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise


    def add_embedding(self):
        with tf.variable_scope("feature_lookup"):
            self.word_embedding_matrix = random_uniform_initializer(self.word_embeddings.shape, "word_embedding_matrix",
                                                                    0.01, trainable=True)

            self.char_embedding_matrix = random_uniform_initializer(self.char_embeddings.shape, "char_embedding_matrix",
                                                                    0.01, trainable=True)

            word_context_embeddings = tf.nn.embedding_lookup(self.word_embedding_matrix, self.word_input_placeholder)
            char_context_embeddings = tf.nn.embedding_lookup(self.char_embedding_matrix, self.char_input_placeholder)

            # temp = tf.cond(self.is_training, True, False)
            if self.config.add_gaussian_noise is True and self.is_training is not None:
                word_context_embeddings = tf.cond(self.is_training,
                                                  lambda: self.gaussian_noise_layer(word_context_embeddings, 0.1),
                                                  lambda: tf.identity(word_context_embeddings))
                char_context_embeddings = tf.cond(self.is_training,
                                                  lambda: self.gaussian_noise_layer(char_context_embeddings, 0.01),
                                                  lambda: tf.identity(char_context_embeddings))

            # [N, H, W] -> [N, H, W, C]
            word_context_embeddings_expanded = tf.expand_dims(word_context_embeddings, -1)
            char_context_embeddings_expanded = tf.expand_dims(char_context_embeddings, -1)

        return word_context_embeddings, word_context_embeddings_expanded, char_context_embeddings, char_context_embeddings_expanded


    """
    def add_prediction_op(self):
        print "***Building network with ReLU activation***"
        word_context_embeddings, word_context_embeddings_expanded, \
        char_context_embeddings, char_context_embeddings_expanded = self.add_embedding()

        # step - 1 :: CNN over characters
        pooled_char_outputs = []
        # char CNN
        for i, char_filter_size in enumerate(self.config.char_filter_sizes):
            with tf.variable_scope("char-conv-maxpool-%s" % char_filter_size):
                # Convolution Layer
                filter_shape = [1, char_filter_size, self.config.char_embedding_dim, 1,
                                self.config.char_num_filters]  # [H, W, in_c, out_c]

                # try xavior also
                # filter = random_truncated_normal_initializer(filter_shape, "filter", stddev=0.1)
                filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.char_num_filters]), name="conv_bias")

                conv = tf.nn.conv3d(
                    char_context_embeddings_expanded,
                    filter,
                    strides=[1, 1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                print "conv shape:", conv.get_shape().as_list()
                h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="relu")  # [B, new_H, new_W, out_c]

                # h_batch_norm = tf.contrib.layers.batch_norm(h,
                #                                   center=True, scale=True,
                #                                   is_training=self.is_training,
                #                                   scope='bn')
                h_shape = h.get_shape().as_list()
                h_4d = tf.reshape(h, [-1, h_shape[2], h_shape[3], h_shape[4]], "char_4d_h")

                pooled = tf.nn.max_pool(
                    h_4d,
                    ksize=[1, self.config.max_word_len - char_filter_size + 1, 1, 1],  # why k_size[2] = 1?
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_shape = pooled.get_shape().as_list()
                pooled_5d = tf.reshape(pooled,
                                       [-1, h_shape[1], pooled_shape[1], pooled_shape[2], pooled_shape[3]])

                pooled_char_outputs.append(pooled_5d)

        char_num_filters_total = self.config.char_num_filters * len(self.config.char_filter_sizes)
        self.h_pool_char = tf.concat(pooled_char_outputs, 4)  # collect across all output channels [B, T, o_h, o_w, o_c]
        self.h_pool_char_flat = tf.reshape(self.h_pool_char, [-1, self.config.max_seq_len,
                                                              char_num_filters_total])  # [B, T, num_features]

        # Step -2 :: Highway layer(s) over char-CNN
        if self.config.use_highway_layer:
            print("***Adding Highway Layer on top of char CNN***")
            curr_input = self.h_pool_char_flat
            for i in range(self.config.num_highway_layers):
                curr_input_2d = tf.reshape(curr_input, [-1, char_num_filters_total])
                # Highway Layer
                with tf.variable_scope("highway_layer_" + str(i + 1)):
                    with tf.variable_scope("transform_gate"):  # use negative bias = -1 (ref: paper, blogs)
                        W_T = xavier_initializer((char_num_filters_total, char_num_filters_total), "W_T")
                        # b_T = xavier_initializer((char_num_filters_total,), "bias_T")
                        b_T = tf.Variable(tf.constant(-2., shape=[char_num_filters_total, ]), name="bias_T")
                        activations_T = tf.nn.sigmoid(
                            tf.nn.xw_plus_b(curr_input_2d, W_T, b_T, name="transform_activations"))
                        print("transformed activations shape: {}".format(activations_T.get_shape().as_list()))

                    with tf.variable_scope("output_gate"):
                        W = xavier_initializer((char_num_filters_total, char_num_filters_total), "W")
                        b = xavier_initializer((char_num_filters_total,), "bias")
                        activations_output = tf.nn.relu(tf.nn.xw_plus_b(curr_input_2d, W, b,
                                                                        name="out_activations"))
                        print("output activations shape: {}".format(activations_output.get_shape().as_list()))

                    activations_carry = 1. - activations_T
                    highway_output = activations_T * activations_output + activations_carry * curr_input_2d
                    curr_input = highway_output
                    self.highway_output = tf.reshape(highway_output, [-1, self.config.max_seq_len,
                                                                      char_num_filters_total], name="highway_output")

        pooled_char_cnn_word_outputs = []
        if self.config.use_highway_layer:
            char_cnn_word_input = tf.expand_dims(self.highway_output, -1)
        else:
            char_cnn_word_input = tf.expand_dims(self.h_pool_char_flat, -1)

        # Step-3: word-CNN over representation obtained from char-CNN + optional Highway layer(s)
        for i, word_filter_size in enumerate(self.config.word_filter_sizes):
            with tf.variable_scope("char-CNN-word-conv-maxpool-%s" % word_filter_size):
                # Convolution Layer
                filter_shape = [word_filter_size, char_num_filters_total, 1,
                                self.config.word_num_filters]  # [H, W, in_c, out_c]

                # try xavior also
                # filter = random_truncated_normal_initializer(filter_shape, "filter", stddev=0.1)
                filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.word_num_filters]), name="conv_bias")

                conv = tf.nn.conv2d(
                    char_cnn_word_input,
                    filter,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                print "conv shape:", conv.get_shape().as_list()
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # [B, new_H, new_W, out_c]

                # h_batch_norm = tf.contrib.layers.batch_norm(h,
                #                                   center=True, scale=True,
                #                                   is_training=self.is_training,
                #                                   scope='bn')

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.max_seq_len - word_filter_size + 1, 1, 1],  # why k_size[2] = 1?
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_char_cnn_word_outputs.append(pooled)

        # Combine all the pooled features
        word_num_filters_total = self.config.word_num_filters * len(self.config.word_filter_sizes)
        self.h_pool_char_cnn_word = tf.concat(pooled_char_cnn_word_outputs,
                                              3)  # collect across all output channels  [B, T, out_H, out_d]
        self.h_pool_char_cnn_word_2d = tf.reshape(self.h_pool_char_cnn_word,
                                                  [-1, word_num_filters_total])  # [B, num_features]

        # Step - 4: word-CNN over pre-trained embeddings
        pooled_word_outputs = []

        # word CNN
        for i, word_filter_size in enumerate(self.config.word_filter_sizes):
            with tf.variable_scope("word-conv-maxpool-%s" % word_filter_size):
                # Convolution Layer
                filter_shape = [word_filter_size, self.config.embedding_dim, 1,
                                self.config.word_num_filters]  # [H, W, in_c, out_c]

                # try xavior also
                # filter = random_truncated_normal_initializer(filter_shape, "filter", stddev=0.1)
                filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.word_num_filters]), name="conv_bias")

                conv = tf.nn.conv2d(
                    word_context_embeddings_expanded,
                    filter,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                print "conv shape:", conv.get_shape().as_list()
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # [B, new_H, new_W, out_c]

                # h_batch_norm = tf.contrib.layers.batch_norm(h,
                #                                   center=True, scale=True,
                #                                   is_training=self.is_training,
                #                                   scope='bn')

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.max_seq_len - word_filter_size + 1, 1, 1],  # why k_size[2] = 1?
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_word_outputs.append(pooled)

        # Combine all the pooled features
        word_num_filters_total = self.config.word_num_filters * len(self.config.word_filter_sizes)
        self.h_pool_word = tf.concat(pooled_word_outputs,
                                     3)  # collect across all output channels  [B, T, out_H, out_d]
        self.h_pool_word_2d = tf.reshape(self.h_pool_word, [-1, word_num_filters_total])  # [B, num_features]

        # Step-5: concat features from both word CNN's
        self.batch_vectors = tf.concat([self.h_pool_char_cnn_word_2d, self.h_pool_word_2d],
                                       1)  # [B, num_features]

        feature_vec_len = self.batch_vectors.get_shape().as_list()[1]
        activations = self.batch_vectors

        # Step-6: FC layer + dropout
        if self.config.use_fc_layer:
            # Final (unnormalized) scores and predictions
            with tf.variable_scope("fc_layer"):
                W = xavier_initializer((feature_vec_len, self.config.fc_layer_dim), "W")
                b = xavier_initializer((self.config.fc_layer_dim,), "bias")
                activations = tf.nn.dropout(
                    tf.nn.relu(tf.nn.xw_plus_b(self.batch_vectors, W, b, name="prediction_logits")),
                    keep_prob=self.dropout_placeholder_fc)
            feature_vec_len = activations.get_shape().as_list()[1]

        # Step-7: softmax layer
        with tf.variable_scope("output_layer"):
            W1 = xavier_initializer((feature_vec_len, self.config.num_classes), "W")
            b1 = xavier_initializer((self.config.num_classes,), "bias")
            predictions = tf.nn.xw_plus_b(activations, W1, b1, name="prediction_logits")

        return predictions
    """


    def add_prediction_op(self):
        print "***Building network with ReLU activation***"
        word_context_embeddings, word_context_embeddings_expanded, \
        char_context_embeddings, char_context_embeddings_expanded = self.add_embedding()

        # step - 1 :: CNN over characters
        pooled_char_outputs = []
        # char CNN
        for i, char_filter_size in enumerate(self.config.char_filter_sizes):
            with tf.variable_scope("char-conv-maxpool-%s" % char_filter_size):
                # Convolution Layer
                filter_shape = [1, char_filter_size, self.config.char_embedding_dim, 1,
                                self.config.char_num_filters]  # [H, W, in_c, out_c]

                # try xavior also
                # filter = random_truncated_normal_initializer(filter_shape, "filter", stddev=0.1)
                filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.char_num_filters]), name="conv_bias")

                conv = tf.nn.conv3d(
                    char_context_embeddings_expanded,
                    filter,
                    strides=self.config.char_stride,
                    padding="VALID",
                    name="conv")
                print "conv shape:", conv.get_shape().as_list()
                h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="relu")  # [B, new_H, new_W, out_c]

                # h_batch_norm = tf.contrib.layers.batch_norm(h,
                #                                   center=True, scale=True,
                #                                   is_training=self.is_training,
                #                                   scope='bn')
                h_shape = h.get_shape().as_list()
                h_4d = tf.reshape(h, [-1, h_shape[2], h_shape[3], h_shape[4]], "char_4d_h")

                pooled = tf.nn.max_pool(
                    h_4d,
                    ksize=[1, h_4d.get_shape().as_list()[1], 1, 1],  # why k_size[2] = 1?
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_shape = pooled.get_shape().as_list()
                pooled_5d = tf.reshape(pooled,
                                       [-1, h_shape[1], pooled_shape[1], pooled_shape[2], pooled_shape[3]])

                pooled_char_outputs.append(pooled_5d)

        char_num_filters_total = self.config.char_num_filters * len(self.config.char_filter_sizes)
        self.h_pool_char = tf.concat(pooled_char_outputs, 4)  # collect across all output channels [B, T, o_h, o_w, o_c]
        self.h_pool_char_flat = tf.reshape(self.h_pool_char, [-1, self.config.max_seq_len,
                                                              char_num_filters_total])  # [B, T, num_features]

        # Step -2 :: Highway layer(s) over char-CNN
        if self.config.use_highway_layer:
            print("***Adding Highway Layer on top of char CNN***")
            curr_input = self.h_pool_char_flat
            for i in range(self.config.num_highway_layers):
                curr_input_2d = tf.reshape(curr_input, [-1, char_num_filters_total])
                # Highway Layer
                with tf.variable_scope("highway_layer_" + str(i + 1)):
                    with tf.variable_scope("transform_gate"):  # use negative bias = -1 (ref: paper, blogs)
                        W_T = xavier_initializer((char_num_filters_total, char_num_filters_total), "W_T")
                        # b_T = xavier_initializer((char_num_filters_total,), "bias_T")
                        b_T = tf.Variable(tf.constant(-2., shape=[char_num_filters_total, ]), name="bias_T")
                        activations_T = tf.nn.sigmoid(
                            tf.nn.xw_plus_b(curr_input_2d, W_T, b_T, name="transform_activations"))
                        print("transformed activations shape: {}".format(activations_T.get_shape().as_list()))

                    with tf.variable_scope("output_gate"):
                        W = xavier_initializer((char_num_filters_total, char_num_filters_total), "W")
                        b = xavier_initializer((char_num_filters_total,), "bias")
                        activations_output = tf.nn.relu(tf.nn.xw_plus_b(curr_input_2d, W, b,
                                                                        name="out_activations"))
                        print("output activations shape: {}".format(activations_output.get_shape().as_list()))

                    activations_carry = 1. - activations_T
                    highway_output = activations_T * activations_output + activations_carry * curr_input_2d
                    curr_input = highway_output
                    self.highway_output = tf.reshape(highway_output, [-1, self.config.max_seq_len,
                                                                      char_num_filters_total], name="highway_output")

        pooled_char_cnn_word_outputs = []
        if self.config.use_highway_layer:
            char_word_context_embeddings = self.highway_output
        else:
            char_word_context_embeddings = self.h_pool_char_flat

        # Step-3: concat features
        self.context_embeddings = tf.expand_dims(tf.concat([word_context_embeddings, char_word_context_embeddings],
                                                           2), -1)
        feature_vec_len = self.context_embeddings.get_shape().as_list()[2]

        # Step - 4: word-CNN
        pooled_word_outputs = []

        # word CNN
        for i, word_filter_size in enumerate(self.config.word_filter_sizes):
            with tf.variable_scope("word-conv-maxpool-%s" % word_filter_size):
                # Convolution Layer
                filter_shape = [word_filter_size, feature_vec_len, 1,
                                self.config.word_num_filters]  # [H, W, in_c, out_c]

                # try xavior also
                # filter = random_truncated_normal_initializer(filter_shape, "filter", stddev=0.1)
                filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.word_num_filters]), name="conv_bias")

                conv = tf.nn.conv2d(
                    self.context_embeddings,
                    filter,
                    strides=self.config.word_stride,
                    padding="VALID",
                    name="conv")
                print "conv shape:", conv.get_shape().as_list()
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # [B, new_H, new_W, out_c]

                # h_batch_norm = tf.contrib.layers.batch_norm(h,
                #                                   center=True, scale=True,
                #                                   is_training=self.is_training,
                #                                   scope='bn')

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, h.get_shape().as_list()[1], 1, 1],  # why k_size[2] = 1?
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_word_outputs.append(pooled)

        # Combine all the pooled features
        word_num_filters_total = self.config.word_num_filters * len(self.config.word_filter_sizes)
        self.h_pool_word = tf.concat(pooled_word_outputs,
                                     3)  # collect across all output channels  [B, T, out_H, out_d]
        self.h_pool_word_2d = tf.reshape(self.h_pool_word,
                                         [-1, word_num_filters_total])  # [B, num_features]

        activations = self.h_pool_word_2d
        feature_vec_len = word_num_filters_total
        # Step-5: FC layer + dropout
        if self.config.use_fc_layer:
            # Final (unnormalized) scores and predictions
            with tf.variable_scope("fc_layer"):
                W = xavier_initializer((feature_vec_len, self.config.fc_layer_dim), "W")
                b = xavier_initializer((self.config.fc_layer_dim,), "bias")
                activations = tf.nn.dropout(
                    tf.nn.relu(tf.nn.xw_plus_b(self.h_pool_word_2d, W, b, name="prediction_logits")),
                    keep_prob=self.dropout_placeholder_fc)
            feature_vec_len = activations.get_shape().as_list()[1]

        # Step-6: softmax layer
        with tf.variable_scope("output_layer"):
            W1 = xavier_initializer((feature_vec_len, self.config.num_classes), "W")
            b1 = xavier_initializer((self.config.num_classes,), "bias")
            predictions = tf.nn.xw_plus_b(tf.nn.dropout(activations, keep_prob=self.dropout_placeholder_word), W1, b1,
                                          name="prediction_logits")

        return predictions


    def l2_loss_sum(self, tvars):
        return tf.add_n([tf.nn.l2_loss(t) for t in tvars], "l2_norms_sum")


    def write_predictions(self, pred, path):
        predicted_label_ids = tf.argmax(tf.nn.softmax(pred), axis=1)
        predicted_labels = map(lambda x: self.idx2label[x], predicted_label_ids)
        np.savetxt(path, predicted_labels)


    def add_loss_op(self, pred):
        tvars = tf.trainable_variables()
        without_bias_tvars = [tvar for tvar in tvars if 'bias' not in tvar.name]

        with tf.variable_scope("loss"):
            # weighted_labels = self.labels_placeholder * self.class_weights
	    weight_per_instance = tf.transpose(tf.matmul(self.labels_placeholder, tf.transpose(self.class_weights))) #shape [1, batch_size]
	    
            weighted_cross_entropy_loss = tf.reduce_mean(tf.multiply(weight_per_instance, tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels_placeholder, logits=pred), name="batch_xentropy_loss"))

            l2_loss = tf.multiply(self.config.reg_val, self.l2_loss_sum(without_bias_tvars), name="l2_loss")
            loss = tf.add(weighted_cross_entropy_loss, l2_loss, name="total_batch_loss")

        tf.summary.scalar("batch_loss", loss)
        return loss


    def add_accuracy_op(self, pred):
        with tf.variable_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1),
                                                       tf.argmax(self.labels_placeholder, axis=1)), dtype=tf.float32),
                                      name="curr_batch_accuracy")
        return accuracy


    def get_prediction_labels(self, pred):
        predicted_ids = tf.argmax(pred, axis=1)
        target_ids = tf.argmax(self.labels_placeholder, axis=1)
        return predicted_ids, target_ids


    def add_training_op(self, loss):
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        with tf.variable_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr, name="adam_optimizer")
            tvars = tf.trainable_variables()
            grad_tvars = optimizer.compute_gradients(loss, tvars)
            self.write_gradient_summaries(grad_tvars)
            train_op = optimizer.apply_gradients(grad_tvars)

        return train_op


    def train_on_batch(self, sess, inputs_batch, labels_batch, merged):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     keep_prob_word=self.config.keep_prob, keep_prob_fc=self.config.keep_prob_fc,
                                     is_training=True)
        _, summary, loss = sess.run([self.train_op, merged, self.loss], feed_dict=feed)
        return summary, loss


    def write_gradient_summaries(self, grad_tvars):
        with tf.variable_scope("gradient_summaries"):
            for (grad, tvar) in grad_tvars:
                mean = tf.reduce_mean(grad)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(grad - mean)))
                tf.summary.histogram("{}/hist".format(tvar.name), grad)
                tf.summary.scalar("{}/mean".format(tvar.name), mean)
                tf.summary.scalar("{}/stddev".format(tvar.name), stddev)
                tf.summary.scalar("{}/sparsity".format(tvar.name), tf.nn.zero_fraction(grad))


    def run_epoch(self, sess, config, dataset, train_writer, merged):
        prog = Progbar(target=1 + len(dataset.train_inputs[0]) / config.batch_size)
        for i, (train_x, train_y) in enumerate(get_minibatches([dataset.train_inputs, dataset.train_targets],
                                                               config.batch_size, is_multi_feature_input=True)):
            print "word input, char input, outout: {}, {}, {}".format(np.array(train_x[0]).shape,
                                                                      np.array(train_x[1]).shape,
                                                                      np.array(train_y).shape)

            summary, loss = self.train_on_batch(sess, train_x, train_y, merged)
            prog.update(i + 1, [("train loss", loss)])

        # feed = self.create_feed_dict(dataset.train_inputs, labels_batch=dataset.train_targets,
        #                              keep_prob_word=self.config.keep_prob, keep_prob_fc=self.config.keep_prob_fc,
        #                              is_training=False)
        # train_accuracy = sess.run(self.accuracy, feed_dict=feed)
        # print "- train Accuracy: {:.2f}".format(train_accuracy * 100.0)

        return summary, loss  # returns for Last batch


    def run_valid_epoch(self, sess, dataset, valid_writer, merged):
        print "\nEvaluating on dev set",
        total_correct = 0
        total_loss = []
        predicted_ids = []
        target_ids = []

        for i, (valid_x, valid_y) in enumerate(
                get_minibatches([dataset.valid_inputs, dataset.valid_targets], self.config.test_batch_size,
                                is_multi_feature_input=True)):
            batch_feed = self.create_feed_dict(valid_x, labels_batch=valid_y, is_training=False)
            batch_predictions, batch_loss, batch_accuracy = sess.run([self.pred, self.loss, self.accuracy],
                                                                     feed_dict=batch_feed)

            total_correct += (batch_accuracy * len(valid_y))
            total_loss.append(batch_loss)

            predicted_ids.extend(list(np.argmax(batch_predictions, axis=1)))
            target_ids.extend(list(np.argmax(valid_y, axis=1)))

        valid_loss = sum(total_loss) / float(len(total_loss))
        valid_accuracy = float(total_correct) / len(dataset.valid_targets)

        valid_f1_score = get_weighted_f1_score(predicted_ids, target_ids)

        return valid_loss, valid_accuracy, valid_f1_score


    def dump_incorrect_predictions(self, predictions, targets, offset, incorrect_predictions_path, predictions_path):
        predicted_label_ids = np.argmax(predictions, axis=1)
        predicted_labels = map(lambda x: self.idx2label[x], predicted_label_ids)

        # dump predictions
        f = open(predictions_path, "a")
        for each in predicted_labels:
            f.write(each + "\n")
        f.close()

        # dump incorrect predictions
        actual_label_ids = np.argmax(targets, axis=1)
        actual_labels = map(lambda x: self.idx2label[x], actual_label_ids)
        write_wrong_predictions(actual_labels, predicted_labels, offset, incorrect_predictions_path)


    def run_test_epoch(self, sess, dataset):
        print "\nEvaluating on Test set",
        total_correct = 0
        total_loss = []
        predicted_ids = []
        target_ids = []

        incorrect_predictions_path = os.path.join(DataConfig.data_dir_path, DataConfig.model_dir,
                                                  DataConfig.test_incorrect_predictions_file)
        predictions_path = os.path.join(DataConfig.data_dir_path, DataConfig.model_dir,
                                        DataConfig.test_predictions_file)

        if os.path.exists(predictions_path):
            os.remove(predictions_path)
        if os.path.exists(incorrect_predictions_path):
            os.remove(incorrect_predictions_path)

        offset = 0

        for i, (test_x, test_y) in enumerate(
                get_minibatches([dataset.test_inputs, dataset.test_targets], self.config.test_batch_size, shuffle=False,
                                is_multi_feature_input=True)):
            batch_feed = self.create_feed_dict(test_x, labels_batch=test_y, is_training=False)
            batch_predictions, batch_loss, batch_accuracy = sess.run([self.pred, self.loss, self.accuracy],
                                                                     feed_dict=batch_feed)

            total_correct += (batch_accuracy * len(test_y))
            total_loss.append(batch_loss)

            predicted_ids.extend(list(np.argmax(batch_predictions, axis=1)))
            target_ids.extend(list(np.argmax(test_y, axis=1)))

            self.dump_incorrect_predictions(batch_predictions, test_y, offset, incorrect_predictions_path,
                                            predictions_path)
            offset += len(test_y)

        test_loss = sum(total_loss) / float(len(total_loss))
        test_accuracy = float(total_correct) / len(dataset.test_targets)

        test_f1_score = get_weighted_f1_score(predicted_ids, target_ids)

        return test_loss, test_accuracy, test_f1_score

        # predicted_ids, target_ids = sess.run([self.prediction_labels, self.target_labels], feed_dict=feed)
        # print_confusion_matrix(predicted_ids, target_ids, self.class_names, "/home/asjindal/metlife_test_cm.png")

        # return test_loss, test_accuracy


    def fit(self, sess, saver, config, dataset, train_writer, valid_writer, merged, resume_training=False):

        if resume_training:
            _, best_valid_accuracy, best_valid_f1_score = self.run_valid_epoch(sess, dataset, valid_writer, merged)
        else:
            best_valid_accuracy = 0.
            best_valid_f1_score = 0.
	
	print "current best valid_accuracy:{:.2f}\tvalid_f1_score:{:.2f}".format(best_valid_accuracy, best_valid_f1_score)

        for epoch in range(config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)

            summary, loss = self.run_epoch(sess, config, dataset, train_writer, merged)

            if (epoch + 1) % dataset.model_config.run_valid_after_epochs == 0:
                valid_loss, valid_accuracy, valid_f1_score = self.run_valid_epoch(sess, dataset, valid_writer, merged)
                print "\n- valid Accuracy: {:.2f}".format(valid_accuracy * 100.0)
                print "- valid f1-score: {:.2f}".format(valid_f1_score * 100.0)
                print "- valid loss: {:.4f}".format(valid_loss)

                valid_accuracy_summary = tf.summary.scalar("valid_UAS", tf.constant(valid_accuracy, dtype=tf.float32))
                valid_writer.add_summary(sess.run(valid_accuracy_summary), epoch + 1)

                if self.config.accuracy_metric == "accuracy":
                    if valid_accuracy > best_valid_accuracy:
                        best_valid_accuracy = valid_accuracy
                        if saver:
                            print "New best dev accuracy! Saving model.."
                            saver.save(sess, os.path.join(DataConfig.data_dir_path, DataConfig.model_dir,
                                                          DataConfig.model_name))
                elif self.config.accuracy_metric == "f1_score":
                    if valid_f1_score > best_valid_f1_score:
                        best_valid_f1_score = valid_f1_score
                        if saver:
                            print "New best dev f1-score! Saving model.."
                            saver.save(sess, os.path.join(DataConfig.data_dir_path, DataConfig.model_dir,
                                                          DataConfig.model_name))

            # trainable variables summary -> only for training
            if (epoch + 1) % dataset.model_config.write_summary_after_epochs == 0:
                train_writer.add_summary(summary, global_step=epoch + 1)

        print
