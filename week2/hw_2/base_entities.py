import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReviewDataset(Dataset):
    def __init__(self, data: pd.core.frame.DataFrame, tokenizer, label_2_idx=None):
        super().__init__()
        # load data:
        self.X = None
        self.reviews = data.iloc[:, 0].tolist()
        self.labels = data.iloc[:, 1].tolist()
        # enumerate labels:
        if label_2_idx is None:
            labels = sorted(set(self.labels))
            self.label_2_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_2_idx = label_2_idx
        self.idx_2_label = {v: k for k, v in self.label_2_idx.items()}
        # Tokenize:
        self.tokenizer = tokenizer
        self.make_all_tokens()

    def make_all_tokens(self):
        """
        A function that created tokens for all the reviews
        :return:
        """
        self.X = np.zeros(shape=(len(self.reviews), self.tokenizer.vector_size))
        N = len(self.reviews)
        for ii, review in enumerate(self.reviews):
            print(f'tokenizing: {ii+1}/{N}|{int(100*ii/N)*"="}{int(100*(N-ii)/N)*"-"}|', end='\r')
            self.X[ii, :] = self.make_review_token(self.tokenizer, review)
        print("")
        self.X = torch.tensor(self.X, dtype=torch.float32)
        return

    @staticmethod
    def make_review_token(tokenizer, review):
        """
        A static method which tokenizes a single review, it may be accessed by either the class instance or
        as a service for a new external review
        :param tokenizer: the tokenizer which was used to train the reviews
        :param review: a string which represents the review
        :return:  tokenized review as a float vector
        """
        review = [word.strip('.,;!@#$%^&*()/"\'<>~') for word in review.lower().split() if len(word) > 1]
        tokens = []
        for word in review:
            if word not in tokenizer.key_to_index:
                continue
            tokens.append(tokenizer[word])
        feat_array = np.concatenate(tokens, axis=0).reshape(-1, tokenizer.vector_size)
        X_i = feat_array.mean(0)
        return X_i

    def __getitem__(self, ii):
        X_i = self.X[ii, :]
        label = self.labels[ii]
        label = self.label_2_idx[label]
        data = {"input_vectors": X_i, "labels": label}
        return data

    def __len__(self):
        return self.X.shape[0]


class GloveClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer1 = nn.Linear(hidden_dim, hidden_dim*2)
        self.hidden_layer2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, X):
        X = self.input_layer(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.hidden_layer1(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.hidden_layer2(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.output_layer(X)
        return X