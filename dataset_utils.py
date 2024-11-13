import os
import numpy as np


def load_imdb_dataset(num_words=None, shuffle=False):
    """
    An adapted & simplified version of keras.datasets.imdb.load_data(),
    which enables quick usage with PyTorch framework only.

    Args:
        num_words (Optional): integer. Only use top #num_words tokens sorted by frequency of occurrence.
        shuffle (Optional): boolean. Whether or not shuffle the dataset.

    Returns:
        Tuple of Numpy arrays: "(x_train, y_train), (x_test, y_test)".
        x_train, x_test: Numpy array of lists.
        y_train, y_test: Numpy array of integers.
    """
    start_char = 1  # special label for the start of a sentence
    oov_char = 2  # special label for out-of-vocabulary
    index_from = 3

    path = "./dataset/imdb.npz"
    if os.path.isfile(path) and os.access(path, os.R_OK):
        with np.load(path, allow_pickle=True) as f:
            x_train, labels_train = f["x_train"], f["y_train"]
            x_test, labels_test = f["x_test"], f["y_test"]
    else:
        raise Exception("""Fail to load IMDB dataset: file not exist or not readable!\n
                           You can download dataset (imdb.npz) from
                           https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz\n
                           Then put it in the \"./dataset\" folder.""")

    if shuffle:
        seed = 2024  # fix the seed for reproduction
        rng = np.random.RandomState(seed)

        indices = np.arange(len(x_train))
        rng.shuffle(indices)
        x_train = x_train[indices]
        labels_train = labels_train[indices]

        indices = np.arange(len(x_test))
        rng.shuffle(indices)
        x_test = x_test[indices]
        labels_test = labels_test[indices]

    # add start_char in the beginning, plus each word's label by index_from
    x_train = [[start_char] + [w + index_from for w in x] for x in x_train]
    x_test = [[start_char] + [w + index_from for w in x] for x in x_test]

    xs = x_train + x_test
    labels = np.concatenate([labels_train, labels_test])

    if not num_words:
        num_words = max(max(x) for x in xs)

    # mark all w >= num_words as oov_char
    xs = [[w if w < num_words else oov_char for w in x] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx], dtype=object), labels[:idx]
    x_test, y_test = np.array(xs[idx:], dtype=object), labels[idx:]
    return (x_train, y_train), (x_test, y_test)
