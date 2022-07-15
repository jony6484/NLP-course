import torch
from torch.utils.data import Dataset
import pandas as pd
import gensim
import argparse
import os
import yaml
from box import Box
from consts import *
from pathlib import Path
from week4.hw3.dataloading import TweetDataset


def main():
    parser = argparse.ArgumentParser(description='Train an LSTM model on the IMDB dataset.')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to YAML config file. Defualt: config.yaml')
    args = parser.parse_args()
    #
    with open(args.config) as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))
    #


    file_path = Path.cwd() / 'week4' / 'hw3' / 'data' / ('train.csv')
    df = pd.read_csv(file_path)
    # dataset = TweetDataset(file_path=file_path, )
    return


if __name__ == '__main__':
    main()