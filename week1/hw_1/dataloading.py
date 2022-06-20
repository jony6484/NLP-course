# from consts import *
from scipy.sparse import csr_matrix
from os import path
import numpy as np


def load_raw_dataset(split, test=False):
    """
    The function is getting  file name, reading the file and returns as dataset
    :param split: name of the file
    :param test: if the file is a test file
    :return: raw dataset as list of lists - each sub list represent sentence
    """
    file_path = path.join('.','data',f'{split}.txt')
    with open(file_path, 'r') as file:
        raw_lines = [line.rstrip().split() for line in file.readlines()]
    sentences = []
    words, pos, labels = [], [], []
    for line in raw_lines:
        if len(line) > 0:
            words.append(line[0])
            pos.append(line[1])
            if not test:
                labels.append(line[-1])
        else:
            last_pos = pos[:-1]
            last_pos.insert(0, 'XX')
            next_pos = pos[1:]
            next_pos.append('XX')
            if not test:
                sentences.append([words, pos, last_pos, next_pos, labels])
            else:
                sentences.append([words, pos, last_pos, next_pos])
            words, pos, labels = [], [], []
    return sentences


def convert_raw_to_features(sentences, feature_maps, test=False):
    """
    The function is getting the raw dataset and creates features for each sentence.
    :param sentences: The raw dataset
    :param feature_maps: list of function for creating the features
    :param test: if the file is a test file
    :return: X- Features, y- labels
    """
    y = []
    X = []
    m = len(feature_maps)
    for sentence in sentences:
        X_i = [feature(sentence) for feature in feature_maps]
        X += list(zip(*X_i))
        if not test:
            y += sentence[-1]
    X = np.array(X).astype(float)
    if not test:
        y = np.array(y)
        return X, y
    else:
        return X


def get_dataset():
    pass
