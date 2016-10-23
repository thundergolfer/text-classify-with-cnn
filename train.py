
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import manage_data
from text_network import TextNetwork
from tensorflow.contrib import learn


# TODO   Initialize the embeddings with pre-trained word2vec vectors. To make this work you need to use 300-dimensional embeddings and initialize them with the pre-trained values.
# TODO   Constrain the L2 norm of the weight vectors in the last layer, just like the original paper. You can do this by defining a new operation that updates the weight values after each training step.
# TODO   Add histogram summaries for weight updates and layer actions and visualize them in TensorBoard.
# TODO   Add L2 regularization to the network to combat overfitting, also experiment with increasing the dropout rate. (The code on Github already includes L2 regularization, but it is disabled by default)
# TODO   Implement cross-validation for the data


# ### Set Parameters


def print_list_params( FLAGS ):
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")


# Training dataset
tf.flags.DEFINE_string("dataset_option",        "movies", "Which dataset to train on (default: movies polarity set)")

# Model Hyperparams
tf.flags.DEFINE_integer("embedding_dim",        128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes",          "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters",          128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob",      0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda",          0.0, "L2 regularizaion lambda (default: 0.0)") # Experiment with this. 0.0 is sub-optimal

# Training parameters
tf.flags.DEFINE_integer("batch_size",           64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs",           200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every",       100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every",     100, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print_list_params( FLAGS )


# ### Data Preparation

## Load data
if FLAGS.dataset_option and FLAGS.dataset_option == "products":
    # load product review dataset
    print("Loading product review data")
    x_text, y = manage_data.load_customer_review_data_and_labels()
else:
    print("Loading movies data...")
    x_text, y = manage_data.load_data_and_labels()

## Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text]) # get the longest sentence
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

## Randomly shuffle data
# We need to seed the random number generator so that we get the same results if we run training twice.
# It is slightly confusing if we receive differing results running the same program twice.
np.random.seed(10)
perm = np.random.permutation(np.arange(len(y)))
x_shuffled = x[perm] # this kind of quick order rearranging is a feature of numpy arrays
y_shuffled = y[perm] # both x and y have now been shuffled identically

# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# ### Training The Model

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextNetwork(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for accuracy and loss
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        # Checkpoint directory will be different for each dataset.
        if FLAGS.dataset_option and FLAGS.dataset_option == "products":
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "product_checkpoints"))
        else:
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch, print_res=True): # You DEFINITELY want print_res False for Juptyer Notebooks
            """
            A single training step
            """
            feed_dict = {cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if print_res:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = manage_data.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch) # TODO investigate this line
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint in {}\n".format(path))
