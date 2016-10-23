import numpy as np
import re
import itertools
from collections import Counter

def sent_cleanup(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    pos_examples = open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines()
    neg_examples = open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines()
    # strip off whitespace from begin and/or end of our sentences
    neg_examples = [sent.strip() for sent in neg_examples]
    pos_examples = [sent.strip() for sent in pos_examples]
    # Split by words
    x_text = pos_examples + neg_examples
    x_text = [sent_cleanup(sent) for sent in x_text]
    # Generate labels
    #   Two-dimensional labelling allow for a measure of both 'positiveness' and 'negativeness' I think
    pos_labels = [[0, 1] for _ in pos_examples]
    neg_labels = [[1, 0] for _ in neg_examples]
    y = np.concatenate([pos_labels, neg_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Creates a batch iterator for our dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = data_size // batch_size + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min( (batch_num + 1) * batch_size, data_size )
            yield data[start_index:end_index] # Code Note: don't create a new list. slower


def reformat_product_file( filepath ):
    """
    Transform Hu and Lui's Customer Review dataset format to two simple lists.
    One list of positive sentences and one list of negative sentences.
    """

    lines = list(open(filename, "r").readlines())
    pos_sents, neg_sents = [], []

    # [t] designates a title. After the title line the sentences begin.
    # the are ended by another [t] title line, or EOF
    past_first_title = False
    for line in lines:
        if line.startswith('[t]'):
            past_first_title = True
            continue # don't bother matching a title line
        if past_first_title:
            # Our pattern looks for sentences (started by '##') that are preceded by
            # positive or negative sentiment declarations, denoted by [-*num*] or [+*num*]
            match = re.match( r'^(.+\[[+-]\d\],?)+##.+', line)
            # The sentence starts after the SECOND '#'
            if '-' in match.group():
                pos_sents.append( line[line.index('#')+2:] )
            elif '+' in match.group():
                neg_sents.append( line[line.index('#')+2:] )

    return pos_sents, neg_sents

def save_formatted_customer_review_dataset( pos_sents, neg_sents ):
    """
    Save the formatted dataset to file so we don't have to reconvert.
    """
    # Check if files don't already exist

    raise NotImplementedError

def load_customer_review_data_and_labels():
    """
    Loads Customer Review data from files, extracts only sentences with sentiment tags,
    splits the data into words, and generates labels.
    Returns split sentences and labels.
    """

    pos_examples, neg_examples = [], []

    # get filename for each product file in
    product_files = ["./data/customer-review-data/Apex-AD2600-Progressive-scan-DVD-player.txt",
                     "./data/customer-review-data/Canon-G3.txt",
                     "./data/customer-review-data/Creative-Labs-Nomad-Jukebox-Zen-Xtra-40GB.txt"
                     "./data/customer-review-data/Nikon-coolpix-4300.txt"
                     "./data/customer-review-data/Nokia-6610.txt"]

    # build 2 lists containing grouping all positive sentences from all files and all negative sentences
    # from all files
    for pf in product_files:
        pos, neg = reformat_product_file( pf )
        pos_examples.extend(pos)
        neg_examples.extend(neg)

    pos_examples = [sent.strip() for sent in pos_examples]
    neg_examples = [sent.strip() for sent in neg_examples]
    # Split by words
    x_text = pos_examples + neg_examples
    x_text = [sent_cleanup(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in pos_examples]
    negative_labels = [[1, 0] for _ in neg_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]
