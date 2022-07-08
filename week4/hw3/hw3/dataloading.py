import torch
from torch.utils.data import Dataset
import pandas as pd
import gensim

from consts import *


class TweetDataset(Dataset):
    def __init__(self, data_args, file_path, vocab=None):
        self.data_args = data_args
        self.file_path = file_path

        # Load data to dataframe
        self.df =

        # Get vocab
        if vocab is None:
            # Tokenize all of the text using gensim.utils.tokenize(text, lowercase=True)
            tokenized_text =
            # Create a set of all the unique tokens in the text
            self.vocab =
        else:
            self.vocab = vocab

        # Add the UNK token to the vocab

        # Set the vocab size
        self.vocab_size =

        # Create a dictionary mapping tokens to indices
        self.token2id =
        self.id2token =

        # Tokenize data using the tokenize function
        self.df[INPUT_IDS] =


    def __len__(self):
        # Return the length of the dataset
        return

    def __getitem__(self, idx):
        # Get the row at idx

        # return the input_ids and the label as tensors, make sure to convert the label type to a long
        return torch.tensor(input_ids), torch.tensor(label, dtype=torch.long)

    def tokenize(self, text):
        input_ids = []
        # Tokenize the text using gensim.utils.tokenize(text, lowercase=True)
        for _ in _:
            # Make sure to trim sequences to max_seq_length

            # Gets the token id, if unknown returns self.unk_token

        return input_ids
