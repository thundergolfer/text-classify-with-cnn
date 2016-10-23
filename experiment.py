
# coding: utf-8

# Use this file to try out the CNN Text Classifying Network on sentences outside the dataset.

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import manage_data
from text_network import TextNetwork
from tensorflow.contrib import learn

tf.flags.DEFINE_string("sent", "", "The sentence to evaluate (default: '')")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# TODO: Refactor this to predict only a single case

if not FLAGS.sent:
    test_sentence = "This is a test sentence!" # Our x data
else:
    test_sentence = FLAGS.sent
x_raw = test_sentence

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating your sentence...\n")

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()

with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = manage_data.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        for x_test_batch in batches:
            prediction = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})

        print("Prediction was: ", prediction[0])

def run_network_on_sentence( sent ):
    raise NotImplementedError


def classification_report( sent ):
    raise NotImplementedError
