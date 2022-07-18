import torch
import random
import numpy as np
import pandas as pd
from consts import *


def set_seed(seed):
    """
    Set random seeds for reproducibility
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_args(training_args):
    """
    This is where you can check the validity of the configuration and set extra attributes that can't be incorporated
    in the YAML file
    """
    return training_args

def save_competitive(test_path, competitive_path, prediction):
    """
    :param test_path: test file location
    :param competitive_path: path to write the competitive file
    :param prediction: model prediction
    :return:
    """
    test_df = pd.read_csv(test_path)
    test_df[LABEL] = prediction
    test_df.to_csv(competitive_path, index=False)


