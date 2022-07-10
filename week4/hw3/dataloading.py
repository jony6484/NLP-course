import torch
from torch.utils.data import Dataset
import pandas as pd
import gensim
import numpy as np
from consts import *


class TweetDataset(Dataset):
    def __init__(self, data_args, file_path, vocab=None):
        self.data_args = data_args
        self.file_path = file_path

        # Load data to dataframe
        self.df = pd.read_csv(file_path)

        # Get vocab
        if vocab is None:
            # Tokenize all of the text using gensim.utils.tokenize(text, lowercase=True)
            tokenized_text = gensim.utils.tokenize(' '.join(self.df['text'].tolist()), lowercase=True)
            # Create a set of all the unique tokens in the text
            self.vocab = set(tokenized_text)
        else:
            self.vocab = vocab

        # Add the UNK token to the vocab
        self.vocab.add(UNK_TOKEN)

        # Set the vocab size
        self.vocab_size = len(self.vocab)

        # Create a dictionary mapping tokens to indices
        self.token2id = {item: val for val, item in enumerate(self.vocab)}
        self.id2token = {v: k for k, v in self.token2id.items()}

        # Tokenize data using the tokenize function
        self.df[INPUT_IDS] = self.df['text'].apply(self.tokenize)


    def __len__(self):
        # Return the length of the dataset
        return self.df.size

    def __getitem__(self, idx):
        # Get the row at idx
        input_ids = self.df.iloc[idx][INPUT_IDS]
        label     = self.df.iloc[idx][label]
        # return the input_ids and the label as tensors, make sure to convert the label type to a long
        return torch.tensor(input_ids), torch.tensor(label, dtype=torch.long)

    def tokenize(self, text):
        input_ids = []
        # Tokenize the text using gensim.utils.tokenize(text, lowercase=True)
        for word in list(gensim.utils.tokenize(text, lowercase=True))[:self.data_args.max_seq_length]:
            # Make sure to trim sequences to max_seq_length

            # Gets the token id, if unknown returns self.unk_token
            if word in self.token2id.keys():
                input_ids.append(self.token2id[word])
            else:
                input_ids.append(self.token2id[UNK_TOKEN])

        return input_ids
