import torch
from torch.utils.data import Dataset
import pandas as pd
import gensim
from gensim import downloader
import numpy as np
from consts import *
from gensim.models import KeyedVectors

class TweetDataset(Dataset):
    def __init__(self, data_args, file_path, vocab=None, is_test=False):
        self.data_args = data_args
        self.file_path = file_path

        # Load data to dataframe
        self.is_test = is_test
        self.df = pd.read_csv(file_path)
        self.embedding_model = downloader.load(data_args.embedding_model)

        # Get vocab
        if vocab is None:
            # Tokenize all of the text using gensim.utils.tokenize(text, lowercase=True)
            tokenized_text = gensim.utils.tokenize(' '.join(self.df['text'].tolist()), lowercase=True)
            # Create a set of all the unique tokens in the text
            self.vocab = set(tokenized_text)
        else:
            self.vocab = vocab

        # Add the UNK token to the vocab
        self.unk_token = len(self.embedding_model)
        self.pad_token = len(self.embedding_model) + 1
        self.vocab.add(UNK_TOKEN)
        self.vocab.add(PAD_TOKEN)

        # Set the vocab size
        self.vocab_size = len(self.vocab)

        # Create a dictionary mapping tokens to indices
        self.token2id = {item: val for val, item in enumerate(self.vocab)}
        self.id2token = {v: k for k, v in self.token2id.items()}

        # Tokenize data using the tokenize function
        self.df[[INPUT_IDS, LENGTH]] = self.df['text'].apply(self.tokenize)

    def __len__(self):
        # Return the length of the dataset
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row at idx
        input_ids = self.df.iloc[idx][INPUT_IDS]
        length    = self.df.iloc[idx][LENGTH]
        if self.is_test:
            return torch.tensor(input_ids), torch.tensor(length)
        label     = self.df.iloc[idx][LABEL]
        # return the input_ids and the label as tensors, make sure to convert the label type to a long
        return torch.tensor(input_ids), torch.tensor(length), torch.tensor(label, dtype=torch.long)

    def tokenize(self, text):
        input_ids = []
        # Tokenize the text using gensim.utils.tokenize(text, lowercase=True)
        # for word in list(gensim.utils.tokenize(text, lowercase=True))[:self.data_args.max_seq_length]:
        for i, token in enumerate(gensim.utils.tokenize(text, lowercase=True)):
            if i >= self.data_args.max_seq_length:
                break
            # Make sure to trim sequences to max_seq_length
            # Gets the token id, if unknown returns self.unk_token
            input_ids.append(self.embedding_model.get_index(token, self.unk_token))
        length = len(input_ids)

        # Pad
        for i in range(self.data_args.max_seq_length - len(input_ids)):
            input_ids.append(self.pad_token)

        return pd.Series([input_ids, length])
