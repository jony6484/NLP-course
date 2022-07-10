from pathlib import Path

PROJECT_NAME = "IMDB_Tutorial"

PROJECT_DIR = Path.home() / "unit 7" / "7.3 Coding Practices"
DATA_DIR = PROJECT_DIR / "data"

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

POSITIVE_KEY = 'pos'
POSITIVE_VALUE = 1.
NEGATIVE_KEY = 'neg'
NEGATIVE_VALUE = 0.

NUM_TRAIN_DEV_INSTANCES = 25000

EPOCH = "epoch"
ITERATION = "iteration"