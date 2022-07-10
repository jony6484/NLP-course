import math

import torch
from torch.utils.data import Dataset
from torchtext.datasets import IMDB
import gensim.utils
from gensim import downloader

from tqdm import tqdm
import pickle

from consts import *


def get_datasets(data_args):
    # Loading from checkpoint
    if data_args.from_checkpoint:
        print(f"Loading datasets from checkpoint {data_args.from_checkpoint}")
        with open(data_args.from_checkpoint, 'rb') as f:
            train_dataset, dev_dataset, test_dataset = pickle.load(f)
        return train_dataset, dev_dataset, test_dataset

    # Load raw datasets
    train_dev_raw_dataset = IMDB(split=TRAIN)
    test_raw_dataset = IMDB(split=TEST)

    # Split train and dev sets
    train_raw_dataset, dev_raw_dataset = train_dev_split(train_dev_raw_dataset, data_args)

    # Load embedding_model for tokenization
    embedding_model = downloader.load(data_args.embedding_model)

    # Create Datasets
    train_dataset = IMDB_Dataset(data_args, TRAIN, train_raw_dataset, embedding_model)
    dev_dataset = IMDB_Dataset(data_args, DEV, dev_raw_dataset, embedding_model)
    test_dataset = IMDB_Dataset(data_args, TEST, test_raw_dataset, embedding_model)

    # Save to checkpoint
    if data_args.to_checkpoint:
        print(f"Saving dataset to checkpoint {data_args.to_checkpoint}")
        with open(data_args.to_checkpoint, 'wb') as f:
            pickle.dump((train_dataset, dev_dataset, test_dataset), f)
    return train_dataset, dev_dataset, test_dataset


class IMDB_Dataset(Dataset):
    def __init__(self, data_args, split, raw_dataset, embedding_model):
        self.data_args = data_args
        self.split = split

        self.positive_key = POSITIVE_KEY
        self.positive_value = POSITIVE_VALUE
        self.negative_key = NEGATIVE_KEY
        self.negative_value = NEGATIVE_VALUE

        self.unk_token = len(embedding_model)
        self.pad_token = len(embedding_model) + 1

        self.labels, self.input_ids, self.lengths = self.preprocess(raw_dataset, split, embedding_model)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        input_ids = self.input_ids[idx]
        length = self.lengths[idx]
        return input_ids, length, label

    def preprocess(self, raw_dataset, split, embedding_model):
        labels, input_ids_list, lengths = [], [], []
        for label, text in tqdm(raw_dataset, desc=f"Tokenizing {split}"):
            if label == self.positive_key:
                labels.append(self.positive_value)
            elif label == self.negative_key:
                labels.append(self.negative_value)
            else:
                raise ValueError(f"Label {label} is unsupported.")
            input_ids, length = self.tokenize(text, embedding_model)
            input_ids_list.append(input_ids)
            lengths.append(length)
        return labels, input_ids_list, lengths

    def tokenize(self, text, embedding_model):
        input_ids = []
        for i, token in enumerate(gensim.utils.tokenize(text, lowercase=True)):
            # Trim sequences to max_seq_length
            if i >= self.data_args.max_seq_length:
                break
            # Gets the token id, if unknown returns self.unk_token
            input_ids.append(embedding_model.get_index(token, self.unk_token))

        # Save length
        length = len(input_ids)

        # Pad
        for i in range(self.data_args.max_seq_length - len(input_ids)):
            input_ids.append(self.pad_token)
        return torch.LongTensor(input_ids), length


def train_dev_split(train_dev_dataset, data_args):
    dev_dataset_size = math.ceil(NUM_TRAIN_DEV_INSTANCES * data_args.dev_frac)
    train_samples, dev_samples = [], []
    for i, sample in enumerate(train_dev_dataset):
        if i < dev_dataset_size:
            dev_samples.append(sample)
        else:
            train_samples.append(sample)
    return train_samples, dev_samples
