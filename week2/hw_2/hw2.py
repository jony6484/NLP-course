import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
from base_entities import ReviewDataset, GloveClassifier
from helper_functions import make_model, train_model, load_tokenizer, save_results
from os.path import exists, join

matplotlib.style.use('bmh')


def main():
    data_train = pd.read_csv('IMDB_train.csv')
    data_test = pd.read_csv('IMDB_test.csv')
    token_model_name = 'glove-twitter-200'
    tokenizer = load_tokenizer(token_model_name)
    dataset_train = ReviewDataset(data_train, tokenizer)
    dataset_test = ReviewDataset(data_test, tokenizer, dataset_train.label_2_idx)
    print('Done loading')
    batch_size = 128
    train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
                              #, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(dataset_test, shuffle=False, batch_size=2*batch_size)
                             #, num_workers=2, persistent_workers=True)
    model, loss_fun, optimizer = make_model(input_dim=dataset_train.tokenizer.vector_size, hidden_dim=512,
                                            num_classes=len(dataset_train.label_2_idx))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss, test_loss, train_acc, test_acc, model = train_model(
        train_loader, test_loader, model, loss_fun, optimizer, device, num_epochs=10)
    res_df = pd.DataFrame({'train_loss': train_loss, 'test_loss': test_loss,
                           'train_acc': train_acc, 'test_acc': test_acc})
    save_results(model, res_df)

    return


if __name__ == '__main__':
    main()