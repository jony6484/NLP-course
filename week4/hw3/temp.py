import torch
from torch.utils.data import Dataset
import pandas as pd
import gensim

from consts import *
from pathlib import Path


def main():
    file_path = Path.cwd() / 'week4' / 'hw3' / 'data' / ('train' + '.csv')
    df = pd.read_csv(file_path)
    return


if __name__ == '__main__':
    main()