import sys
import time
import numpy as np
import cPickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve
from sklearn.utils import class_weight
import itertools
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_class_weights_for_imbalance_labels(labels, normalize=False):
    unique_labels = list(np.unique(labels))
    new_weight = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

    new_weight = map(lambda x: 1. if x < 1. else x, new_weight)
    new_weight = map(lambda x: 2. if x > 2. else x, new_weight)
    if normalize:
        new_weight /= len(new_weight)       # normalization
    return dict(zip(unique_labels, new_weight))


def get_minibatches(data, minibatch_size, shuffle=False, is_multi_feature_input=False):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
        is_multi_feature_input: True if multiple type features are present ex. (word, pos, label)
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    if is_multi_feature_input:
        list_data = type(data) is list and (type(data[0][0]) is list or type(data[0][0]) is np.ndarray)
    else:
        list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    # data_size = len(data[0]) if list_data else len(data)
    data_size = len(data[0]) if not is_multi_feature_input else len(data[0][0])
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        if is_multi_feature_input:
            yield [[minibatch(data[0][i], minibatch_indices) for i in range(len(data[0]))],
                   minibatch(data[1], minibatch_indices)] if list_data \
                else [minibatch(data[0][i], minibatch_indices) for i in range(len(data[0]))]
        else:
            yield [minibatch(d, minibatch_indices) for d in data] if list_data \
                else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def test_all_close(name, actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("{:} failed, expected output to have shape {:} but has shape {:}"
                         .format(name, expected.shape, actual.shape))
    if np.amax(np.fabs(actual - expected)) > 1e-6:
        raise ValueError("{:} failed, expected {:} but value is {:}".format(name, expected, actual))
    else:
        print name, "passed!"


def get_pickle(path):
    data = cPickle.load(open(path, "rb"))
    return data


def dump_pickle(data, path):
    with open(path, "w") as f:
        cPickle.dump(data, f)


def get_vocab_dict(items):
    item2idx = {}
    idx = 0
    for item in items:
        item2idx[item] = idx
        idx += 1
    return item2idx


def logged_loop(iterable, n=None):
    if n is None:
        n = len(iterable)
    step = max(1, n / 1000)
    prog = Progbar(n)
    for i, elem in enumerate(iterable):
        if i % step == 0 or i == n - 1:
            prog.update(i + 1)
        yield elem


def plot_confusion_matrix(cm, classes, save_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)


def print_confusion_matrix(predictions, targets, class_names, save_path):
    cnf_matrix = confusion_matrix(targets, predictions)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, class_names, save_path, title='Confusion matrix')


class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """


    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose


    def update(self, current, values=[], exact=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")


    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)


# glove_dict = get_pickle("/home/asjindal/Work/tf/pkl/glove.6B.100d.pkl")
# print len(glove_dict)
# print glove_dict["apple"], type(glove_dict["apple"])
# print "A"


def write_wrong_predictions(actual_labels, predicted_labels, offset, path):
    flag = False
    if os.path.exists(path):
        flag = True

    f = open(path, "a")
    if not flag:
        f.write("instance_id(1 indexed)\tactual_label\tpredicated_label\n")

    for i, (actual_label, pred_label) in enumerate(zip(actual_labels, predicted_labels)):
        if pred_label != actual_label:
            f.write(str(offset + i + 1) + "\t" + actual_label + "\t" + pred_label + "\n")
    f.close()


def get_weighted_f1_score(actual_labels, predicted_labels):
    return f1_score(actual_labels, predicted_labels, average="weighted")  # scalar value


def get_f1_score_per_class(actual_labels, predicted_labels):
    return f1_score(actual_labels, predicted_labels, average=None)  # 1-d numpy array


def make_embedding_to_pkl():
    senna_file = ("/home/asjindal/Work/Retraining/en-senna-50.txt", "senna.50d_dict.pkl")
    glove_50d_file = ("/home/asjindal/data/glove/glove.6B.50d.txt", "glove.6B.50d_dict.pkl")
    glove_100d_file = ("/home/asjindal/data/glove/glove.6B.100d.txt", "glove.6B.100d_dict.pkl")
    glove_300d_file = ("/home/asjindal/Downloads/glove.42B.300d.txt", "glove.42B.300d_dict.pkl")

    all_files = [senna_file, glove_50d_file, glove_100d_file, glove_300d_file]

    word_vectors = {}
    for file in all_files:
        lines = open(file[0], "r").readlines()
        if "\t" in lines[0].strip():
            delim = "\t"
        else:
            delim = " "
        for line in lines:
            sp = line.strip().split(delim)
            """
            if delim == " ":
                word_vectors[sp[0]] = np.array([float(x) for x in sp[1:]])
            else:
                word_vectors[sp[0]] = np.array([float(x) for x in sp[1].split()])
            """

            if delim == " ":
                word_vectors[sp[0]] = " ".join(sp[1:]).strip()
            else:
                word_vectors[sp[0]] = sp[1].strip()

        print "Loaded!"
        dump_pickle(word_vectors, "/home/asjindal/data/embeddings_pkl/" + file[1])
        print "Done!"


# make_embedding_to_pkl()

# glove300d_pkl = get_pickle("/home/asjindal/data/embeddings_pkl/glove300d_dict.pkl")
# print len(glove300d_pkl)
# print glove300d_pkl["apple"], type(glove300d_pkl["apple"])
