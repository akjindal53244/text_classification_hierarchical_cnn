#! /usr/bin/env python

import tensorflow as tf
import os
import time
from text_cnn import TextCNN
from utils.feature_extraction import load_datasets, DataConfig, Flags
from utils.general_utils import print_confusion_matrix


def highlight_string(temp):
    print 80 * "="
    print temp
    print 80 * "="


def main(flag, load_existing_dump=False):
    highlight_string("INITIALIZING")
    print "loading data.."

    dataset = load_datasets(load_existing_dump)
    config = dataset.model_config

    print "word vocab Size: {}".format(len(dataset.word2idx))
    print "char vocab Size: {}".format(len(dataset.char2idx))
    print "Training data Size: {}".format(len(dataset.train_inputs[0]))
    print "valid data Size: {}".format(len(dataset.valid_inputs[0]))
    print "test data Size: {}".format(len(dataset.test_inputs[0]))

    print "word_vocab, embedding_matrix_size: ", len(dataset.word2idx), len(dataset.word_embedding_matrix)
    print "char_vocab, embedding_matrix_size: ", len(dataset.char2idx), len(dataset.char_embedding_matrix)


    if not os.path.exists(os.path.join(DataConfig.data_dir_path, DataConfig.model_dir)):
        os.makedirs(os.path.join(DataConfig.data_dir_path, DataConfig.model_dir))

    with tf.Graph().as_default(), tf.Session() as sess:
        print "Building network...",
        start = time.time()
        with tf.variable_scope("model") as model_scope:
            model = TextCNN(config, dataset.idx2label, dataset.word_embedding_matrix, dataset.label2idx,
                            dataset.char_embedding_matrix)
            # exit(0)
            saver = tf.train.Saver()

        print "took {:.2f} seconds\n".format(time.time() - start)
	print "Model evaluation metric: {}\n".format(config.accuracy_metric)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(DataConfig.data_dir_path, DataConfig.summary_dir,
                                                          DataConfig.train_summ_dir), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(DataConfig.data_dir_path, DataConfig.summary_dir,
                                                          DataConfig.test_summ_dir))

        if flag == Flags.TRAIN:

            # Variable initialization -> not needed for .restore()
            """ The variables to restore do not have to have been initialized,
            as restoring is itself a way to initialize variables. """
            sess.run(tf.global_variables_initializer())
            """ call 'assignment' after 'init' only, else 'assignment' will get reset by 'init' """
            sess.run(tf.assign(model.word_embedding_matrix, model.word_embeddings))
            sess.run(tf.assign(model.char_embedding_matrix, model.char_embeddings))

            highlight_string("TRAINING")
            model.print_trainable_varibles()
	    
	    resume_training = False
	    
	    # Code for resuming training from previous saved checkpoint
	    if config.resume_training_from_saved_checkpoint:
		ckpt_path = tf.train.latest_checkpoint(os.path.join(DataConfig.data_dir_path, DataConfig.model_dir))
	        if ckpt_path is not None:
		    saver.restore(sess, ckpt_path)
		    print "Resuming training from previous saved checkpoint..."
		    resume_training = True
		else:
		    print "No previous checkpoint found! Starting training..."

            model.fit(sess, saver, config, dataset, train_writer, valid_writer, merged, resume_training = resume_training)

            # Testing
            highlight_string("Testing")
            print "Restoring best found parameters on dev set"
            saver.restore(sess, os.path.join(DataConfig.data_dir_path, DataConfig.model_dir,
                                             DataConfig.model_name))

            test_loss, test_accuracy, test_f1_score = model.run_test_epoch(sess, dataset)
            print "- Test Accuracy: {:.2f}".format(test_accuracy * 100.0)
	    print "- Test f1-score: {:.2f}".format(test_f1_score * 100.0)
            print "- Test loss: {:.4f}".format(test_loss)

            train_writer.close()
            valid_writer.close()

        else:
            ckpt_path = tf.train.latest_checkpoint(os.path.join(DataConfig.data_dir_path,
                                                                DataConfig.model_dir))
            if ckpt_path is not None:
                print "Found checkpoint! Restoring variables.."
                saver.restore(sess, ckpt_path)
                highlight_string("Testing")
                valid_loss, valid_accuracy, valid_f1_score = model.run_valid_epoch(sess, dataset, valid_writer, merged)
                print "- valid Accuracy: {:.2f}".format(valid_accuracy * 100.0)
		print "- valid f1-score: {:.2f}".format(valid_f1_score * 100.0)
                print "- valid loss: {:.4f}".format(valid_loss)
                test_loss, test_accuracy, test_f1_score = model.run_test_epoch(sess, dataset)
                print "- Test Accuracy: {:.2f}".format(test_accuracy * 100.0)
		print "- Test f1-score: {:.2f}".format(test_f1_score * 100.0)
                print "- Test loss: {:.4f}".format(test_loss)
                # print_confusion_matrix
            else:
                print "No checkpoint found!"


if __name__ == '__main__':
    main(Flags.TEST, load_existing_dump=True)

