# from consts import *
from scipy.sparse import csr_matrix
from os import path
import numpy as np

# fix test- without labels
def load_raw_dataset(split):
    file_path = path.join('.','data',f'{split}.txt')
    with open(file_path, 'r') as file:
        raw_lines = [line.rstrip().split() for line in file.readlines()]
    sentences = []
    words, pos, labels = [], [], []
    for line in raw_lines:
        if len(line) > 0:
            words.append(line[0])
            pos.append(line[1])
            labels.append(int(line[2] == 'I'))
        else:
            last_pos = pos[:-1]
            last_pos.insert(0, 'XX')
            next_pos = pos[1:]
            next_pos.append('XX')
            sentences.append([words, pos, last_pos, next_pos, labels])
            words, pos, labels = [], [], []
    return sentences


def convert_raw_to_features(sentences, feature_maps):
    y = []
    X = []
    m = len(feature_maps)
    for sentence in sentences:
        X_i = [feature(sentence) for feature in feature_maps]
        X += list(zip(*X_i))
        y += sentence[-1]
    X = np.array(X)#.astype(float)
    y = np.array(y)
    return X, y


def get_dataset():
    pass