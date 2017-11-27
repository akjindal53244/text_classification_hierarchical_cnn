import os
import numpy as np
import datetime
from enum import Enum
from general_utils import get_pickle, dump_pickle, get_vocab_dict, get_class_weights_for_imbalance_labels

NULL = "<null>"
UNK = "<unk>"
PAD_TOKEN = "<pad>"
PAD_CHAR = "<p>"

today_date = str(datetime.datetime.now().date())

cv_id = "2"


def fst(a):
    assert len(a) == 2
    return a[0]


def snd(a):
    assert len(a) == 2
    return a[1]


class DataConfig:  # data, embedding, model path etc.
    # Data dir
    data_dir_path = "./data"
    dataset_identity = "_metlife"
    architecture_identity = "_char_word_weighted_256_0.0002"

    # train/dev/test data paths
    train_path = "5-fold/" + cv_id + "/data_train"
    valid_path = "5-fold/" + cv_id + "/data_valid"
    test_path = "5-fold/" + cv_id + "/data_test"

    # embedding
    embedding_file = "en-cw.txt"

    # model saver
    model_dir = "params" + dataset_identity + today_date + "_cv" + cv_id + architecture_identity
    model_name = "model.weights"
    test_incorrect_predictions_file = "test_incorrect_predicted.txt"
    test_predictions_file = "test_pred.txt"

    # summary
    summary_dir = model_dir
    train_summ_dir = "train_summaries"
    test_summ_dir = "valid_summaries"

    # dump_only_word_excel - vocab
    dump_dir = "./data/dump" + dataset_identity
    word_vocab_file = "word2idx.pkl"
    char_vocab_file = "char2idx.pkl"
    label_vocab_file = "label2idx.pkl"

    # dump_only_word_excel - embedding
    word_emb_file = "word_emb.pkl"  # 2d array
    char_emb_file = "char_emb.pkl"  # 2d array


class ModelConfig(object):  # Takes care of shape, dimensions used for tf model
    # Input
    word_features_types = None
    char_features_types = None
    num_features_types = None

    # char CNN
    char_filter_sizes = [2, 3, 4]
    char_stride = [1, 1, 1, 1, 1]
    char_num_filters = 16
    char_embedding_dim = 16

    # word CNN
    word_filter_sizes = [2, 3, 4]  # org: [2,3,4]
    word_stride = [1, 1, 1, 1]
    word_num_filters = 128
    embedding_dim = 50

    # Noise
    add_gaussian_noise = False

    # weighted loss for imbalance dataset
    use_weighted_loss = True
    class_weights = None

    # output
    num_classes = -1

    # Vocab
    word_vocab_size = None
    char_vocab_size = None
    label_vocab_size = None

    max_seq_len = None
    max_word_len = None

    # num_epochs
    n_epochs = 100

    # batch_size
    batch_size = 256
    test_batch_size = 4 * batch_size

    # dropout
    keep_prob = 0.5
    reg_val = 1e-5
    keep_prob_fc = 0.7

    # learning_rate
    lr = 0.0002

    # load existing vocab
    load_existing_vocab = False

    # layer specific settings
    use_highway_layer = False
    num_highway_layers = 1

    # fc layer
    use_fc_layer = False
    fc_layer_dim = 200

    # summary
    write_summary_after_epochs = 100

    # valid run
    run_valid_after_epochs = 1

    # evaluation
    accuracy_metric = "f1_score"

    # checkpoints
    resume_training_from_saved_checkpoint = True


class SettingsConfig:  # enabling and disabling features, feature types
    # Features
    use_word = True
    is_lower_word = True
    is_lower_char = True


class Flags(Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3


class Char(object):
    def __init__(self, char):
        self.char = char.lower() if SettingsConfig.is_lower_char else char


class Token(object):
    def __init__(self, word):
        self.word = word.lower() if SettingsConfig.is_lower_word else word
        self.chars = [Char(char) for char in self.word]


    def is_null_token(self):
        if self.word == NULL:
            return True
        return False


    def is_unk_token(self):
        if self.word == UNK:
            return True
        return False


class Sentence(object):
    def __init__(self, tokens, label):
        self.tokens = tokens
        self.label = label


class Dataset(object):
    def __init__(self, model_config, train_data_obj, valid_data_obj, test_data_obj, feature_extractor):
        self.model_config = model_config
        self.train_data_obj = train_data_obj
        self.valid_data_obj = valid_data_obj
        self.test_data_obj = test_data_obj
        self.feature_extractor = feature_extractor

        # Vocab
        self.word2idx = None
        self.idx2word = None

        # Vocab
        self.char2idx = None
        self.idx2char = None

        self.label2idx = None
        self.idx2label = None

        # Embedding Matrix
        self.word_embedding_matrix = None
        self.char_embedding_matrix = None

        # input & outputs
        self.train_inputs, self.train_targets = None, None
        self.valid_inputs, self.valid_targets = None, None
        self.test_inputs, self.test_targets = None, None


    def get_max_seq_len(self):
        input_seq_len_list = []
        input_seq_len_list.extend(map(lambda x: len(x.tokens), self.train_data_obj))
        input_seq_len_list.extend(map(lambda x: len(x.tokens), self.valid_data_obj))
        input_seq_len_list.extend(map(lambda x: len(x.tokens), self.test_data_obj))

        return max(input_seq_len_list)


    def get_max_word_len(self):
        input_word_len_list = []
        input_word_len_list.extend([len(token.chars) for sentence in self.train_data_obj for token in sentence.tokens])
        input_word_len_list.extend([len(token.chars) for sentence in self.test_data_obj for token in sentence.tokens])
        input_word_len_list.extend([len(token.chars) for sentence in self.valid_data_obj for token in sentence.tokens])
        return max(input_word_len_list)


    def build_vocab(self):
        all_words = set()
        all_chars = set()
        all_labels = set()
        train_labels = list()  # to be used for getting class_weights for imbalanced datasets
        flatten = lambda l: [item for sublist in l for item in sublist]
        for sentence in self.train_data_obj:
            all_words.update(set(map(lambda x: x.word, sentence.tokens)))
            all_chars.update(set(flatten(map(lambda x: map(lambda y: y.char, x.chars), sentence.tokens))))
            all_labels.add(sentence.label)
            train_labels.append(sentence.label)
        for sentence in self.valid_data_obj:
            all_words.update(set(map(lambda x: x.word, sentence.tokens)))
            all_chars.update(set(flatten(map(lambda x: map(lambda y: y.char, x.chars), sentence.tokens))))
            all_labels.add(sentence.label)
        for sentence in self.test_data_obj:
            all_words.update(set(map(lambda x: x.word, sentence.tokens)))
            all_chars.update(set(flatten(map(lambda x: map(lambda y: y.char, x.chars), sentence.tokens))))
            all_labels.add(sentence.label)

        all_words.add(PAD_TOKEN)
        all_chars.add(PAD_CHAR)

        word_vocab = list(all_words)
        char_vocab = list(all_chars)
        label_vocab = list(all_labels)

        word2idx = get_vocab_dict(word_vocab)
        idx2word = {idx: word for (word, idx) in word2idx.items()}

        char2idx = get_vocab_dict(char_vocab)
        idx2char = {idx: char for (char, idx) in char2idx.items()}

        label2idx = get_vocab_dict(label_vocab)
        idx2label = {idx: label for (label, idx) in label2idx.items()}

        self.word2idx = word2idx
        self.idx2word = idx2word

        self.char2idx = char2idx
        self.idx2char = idx2char

        self.label2idx = label2idx
        self.idx2label = idx2label


    def build_embedding_matrix(self):

        # load word vectors
        word_vectors = {}
        char_vectors = {}
        embedding_lines = open(os.path.join(DataConfig.data_dir_path, DataConfig.embedding_file), "r").readlines()
        for line in embedding_lines:
            sp = line.strip().split()
            word_vectors[sp[0]] = [float(x) for x in sp[1:]]

        # word embedding
        self.model_config.word_vocab_size = len(self.word2idx)
        self.model_config.char_vocab_size = len(self.char2idx)
        word_embedding_matrix = np.asarray(
            np.random.normal(0, 0.9, size=(self.model_config.word_vocab_size, self.model_config.embedding_dim)),
            dtype=np.float32)

        char_embedding_matrix = np.asarray(
            np.random.normal(0, 0.9, size=(self.model_config.char_vocab_size, self.model_config.char_embedding_dim)),
            dtype=np.float32)

        for (word, idx) in self.word2idx.items():
            if word in word_vectors:
                word_embedding_matrix[idx] = word_vectors[word]
            elif word.lower() in word_vectors:
                word_embedding_matrix[idx] = word_vectors[word.lower()]

        # set zero vector to PADDING token
        word_embedding_matrix[self.word2idx[PAD_TOKEN]] = np.zeros((self.model_config.embedding_dim,), dtype=np.float32)
        char_embedding_matrix[self.char2idx[PAD_CHAR]] = np.zeros((self.model_config.char_embedding_dim,),
                                                                  dtype=np.float32)

        self.word_embedding_matrix = word_embedding_matrix
        self.char_embedding_matrix = char_embedding_matrix


    def convert_data_to_ids(self):
        self.train_inputs, self.train_targets = self.feature_extractor. \
            create_instances_for_data(self.train_data_obj, self.word2idx, self.char2idx, self.label2idx)

        self.model_config.class_weights = [1.] * len(self.idx2label)
        # for setting class weights for imbalanced datasets
        if self.model_config.use_weighted_loss:
            train_label_ids = np.argmax(self.train_targets, axis=1)
            label_idx2_weight_dict = get_class_weights_for_imbalance_labels(train_label_ids)

            train_label_ids = label_idx2_weight_dict.keys()
            for idx in train_label_ids:
                self.model_config.class_weights[idx] = label_idx2_weight_dict[idx]
	    
            print "\n\n******label weights****\n\n"
	    for idx, weight in enumerate(self.model_config.class_weights):
		print "class_label:{}\tweight:{}".format(self.idx2label[idx], weight)
	    print "***************************\n"


        self.valid_inputs, self.valid_targets = self.feature_extractor. \
            create_instances_for_data(self.valid_data_obj, self.word2idx, self.char2idx, self.label2idx)
        self.test_inputs, self.test_targets = self.feature_extractor. \
            create_instances_for_data(self.test_data_obj, self.word2idx, self.char2idx, self.label2idx)


    def add_to_vocab(self, words, prefix=""):
        idx = len(self.word2idx)
        for token in words:
            if prefix + token not in self.word2idx:
                self.word2idx[prefix + token] = idx
                self.idx2word[idx] = prefix + token
                idx += 1


class FeatureExtractor(object):
    def __init__(self, model_config):
        self.model_config = model_config


    def pad_to_max_seq_len(self, word_ids, pad_token_id):
        padded_word_ids = word_ids[:]
        padded_word_ids.extend([pad_token_id] * (self.model_config.max_seq_len - len(word_ids)))
        return padded_word_ids


    def pad_to_max_word_len(self, char_ids, pad_char_id):
        padded_char_ids = char_ids[:]
        padded_char_ids.extend(([pad_char_id] * (self.model_config.max_word_len - len(char_ids))))
        return padded_char_ids


    def create_instances_for_data(self, data_obj, word2idx, char2idx, label2idx):
        labels = []
        word_inputs = []
        char_inputs = []

        pad_token_id = word2idx[PAD_TOKEN]
        pad_char_id = char2idx[PAD_CHAR]

        for i, sentence in enumerate(data_obj):
            word_ids = map(lambda x: word2idx[x.word] if x.word in word2idx else pad_token_id, sentence.tokens)
            padded_word_ids = self.pad_to_max_seq_len(word_ids, pad_token_id)

            char_ids_list = [[char2idx[x.char] if x.char in char2idx else pad_char_id for x in token.chars] for token in
                             sentence.tokens]
            # append for extra tokens
            char_ids_list.extend([[pad_char_id]] * (len(padded_word_ids) - len(word_ids)))
            padded_char_ids_list = [self.pad_to_max_word_len(char_ids, pad_char_id) for char_ids in char_ids_list]

            label_id = label2idx[sentence.label]
            word_inputs.append(padded_word_ids)
            char_inputs.append(padded_char_ids_list)

            labels.append(label_id)

        targets = np.zeros((len(labels), self.model_config.num_classes), dtype=np.int32)
        targets[np.arange(len(targets)), labels] = 1

        return [word_inputs, char_inputs], targets


class DataReader(object):
    def __init__(self):
        print "A"


    def read_instance(self, line):
        label, uttr = line.strip().split("\t")
        words = uttr.split(" ")
        tokens = map(lambda x: Token(x), words)
        sentence = Sentence(tokens, label)
        return sentence


    def read_data(self, data_lines):
        data_objects = []
        for line in data_lines:
            data_objects.append(self.read_instance(line))
        return data_objects


def load_datasets(load_existing_dump=False):
    model_config = ModelConfig()

    data_reader = DataReader()
    train_lines = open(os.path.join(DataConfig.data_dir_path, DataConfig.train_path), "r").readlines()
    valid_lines = open(os.path.join(DataConfig.data_dir_path, DataConfig.valid_path), "r").readlines()
    test_lines = open(os.path.join(DataConfig.data_dir_path, DataConfig.test_path), "r").readlines()

    # Load data
    train_data_obj = data_reader.read_data(train_lines)
    print ("Loaded Train data")
    valid_data_obj = data_reader.read_data(valid_lines)
    print ("Loaded Dev data")
    test_data_obj = data_reader.read_data(test_lines)
    print ("Loaded Test data")

    feature_extractor = FeatureExtractor(model_config)
    dataset = Dataset(model_config, train_data_obj, valid_data_obj, test_data_obj, feature_extractor)
    dataset.model_config.max_seq_len = dataset.get_max_seq_len()
    dataset.model_config.max_word_len = dataset.get_max_word_len()

    # Vocab processing
    if load_existing_dump:
        dataset.word2idx = get_pickle(os.path.join(DataConfig.dump_dir, DataConfig.word_vocab_file))
        dataset.idx2word = {idx: word for (word, idx) in dataset.word2idx.items()}

        dataset.char2idx = get_pickle(os.path.join(DataConfig.dump_dir, DataConfig.char_vocab_file))
        dataset.idx2char = {idx: char for (char, idx) in dataset.char2idx.items()}

        dataset.label2idx = get_pickle(os.path.join(DataConfig.dump_dir, DataConfig.label_vocab_file))
        dataset.idx2label = {idx: label for (label, idx) in dataset.label2idx.items()}

        dataset.model_config.load_existing_vocab = True
        print "loaded existing Vocab!"

        dataset.word_embedding_matrix = get_pickle(os.path.join(DataConfig.dump_dir, DataConfig.word_emb_file))
        dataset.char_embedding_matrix = get_pickle(os.path.join(DataConfig.dump_dir, DataConfig.char_emb_file))
        print "loaded existing embedding matrix!"

    else:
        dataset.build_vocab()
        dump_pickle(dataset.word2idx, os.path.join(DataConfig.dump_dir, DataConfig.word_vocab_file))
        dump_pickle(dataset.char2idx, os.path.join(DataConfig.dump_dir, DataConfig.char_vocab_file))
        dump_pickle(dataset.label2idx, os.path.join(DataConfig.dump_dir, DataConfig.label_vocab_file))

        dataset.model_config.load_existing_vocab = True
        print "Vocab Build Done!"
        dataset.build_embedding_matrix()
        print "embedding matrix Build Done"
        dump_pickle(dataset.word_embedding_matrix, os.path.join(DataConfig.dump_dir, DataConfig.word_emb_file))
        dump_pickle(dataset.char_embedding_matrix, os.path.join(DataConfig.dump_dir, DataConfig.char_emb_file))

    dataset.model_config.num_classes = len(dataset.label2idx)

    print "converting data into ids.."
    dataset.convert_data_to_ids()
    print "Done!"

    return dataset
