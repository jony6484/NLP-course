import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_entities import ReviewDataset, GloveClassifier
import gensim
import gensim.downloader
from gensim.models import KeyedVectors
from os.path import exists, join
from copy import deepcopy
import os

def load_tokenizer(token_model_name='glove-twitter-200'):
    """
    A function to load a trained tokenizer and svae it locally
    :param token_model_name: te name of the trained tokenizer (glove/ word2vec)
    :return: the tokenizer object
    """
    token_model_path = join('.', 'model_data', token_model_name + ".model")
    if not exists(join('.', 'model_data')):
        os.mkdir('model_data')
    if exists(token_model_path):
        tokenizer = KeyedVectors.load(token_model_path)
    else:
        tokenizer = gensim.downloader.load(token_model_name)
        tokenizer.save(token_model_path)
    return tokenizer


def make_model(input_dim, num_classes, hidden_dim):
    """
    A function to instantiate a new model and pack it in the return with the loss function and optimizer
    :param input_dim: the tokenized dim
    :param num_classes: the output dim - how many classes
    :param hidden_dim: control over the size of the network
    :return: model, loss function and optimizer objects
    """
    model = GloveClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.00003)
    return model, loss_fun, optimizer


def train_model(train_loader, test_loader, model, loss_fun, optimizer, device, num_epochs=2):
    """
    A function to train the model
    :param train_loader: train_loader
    :param test_loader: test_loader
    :param model: model
    :param loss_fun: loss_fun
    :param optimizer: optimizer
    :param device: device
    :param num_epochs: num_epochs
    :return: the training states and the model at its best state
    """
    train_loss, test_loss, train_acc, test_acc = [], [], [], []
    best_model = {'test_accuracy': 0, 'epoch': -1, 'net': None}
    Nb = len(iter(train_loader))
    model.to(device)
    for epoch_i in range(num_epochs):
        model.train()
        batch_loss = []
        batch_acc = []
        for ii, batch in enumerate(train_loader):
            X = batch['input_vectors'].to(device)
            y = batch['labels'].to(device)
            logits = model(X)
            loss = loss_fun(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.detach().item())
            batch_acc.append(100*torch.mean((torch.argmax(logits, axis=1) == y).detach().float().cpu()).item())
            print(f'batch:{ii + 1}/{Nb}|{int(100 * ii / Nb) * "="}{int(100 * (Nb - ii) / Nb) * "-"}'
                  f'|loss:{batch_loss[ii]:0.3f}|accuracy:{batch_acc[ii]:0.2f}%', end='\r')
        train_loss.append(np.mean(batch_loss))
        train_acc.append((np.mean(batch_acc)))
        model.eval()
        with torch.no_grad():
            batch_loss = []
            batch_acc = []
            for batch in test_loader:
                X = batch['input_vectors'].to(device)
                y = batch['labels'].to(device)
                logits = model(X)
                loss = loss_fun(logits, y)
                batch_loss.append(loss.detach().item())
                batch_acc.append(100*torch.mean((torch.argmax(logits, axis=1) == y).detach().float().cpu()).item())
            test_loss.append(np.mean(batch_loss))
            test_acc.append((np.mean(batch_acc)))
        if test_acc[-1] > best_model['test_accuracy']:
            best_model['test_accuracy'] = test_acc[-1]
            best_model['epoch'] = epoch_i
            best_model['net'] = deepcopy(model.state_dict())
            checkpoint_text = " *checkpoint*"
        print(f'epoch:{epoch_i + 1}/{num_epochs}|loss Train/Test: {train_loss[epoch_i]:0.3f}/{test_loss[epoch_i]:0.3f}|'
              f'accuracy Train/Test: {train_acc[epoch_i]:0.2f}%/{test_acc[epoch_i]:0.2f}%{checkpoint_text}')
        checkpoint_text = ""
    model.load_state_dict(best_model['net'])
    model.to('cpu')
    return train_loss, test_loss, train_acc, test_acc, model


def save_results(model, res_df):
    """
    A funcition to save the results - model and training stats
    :param model: the trained model
    :param res_df: pandas df with train\test loss and accuracy columns
    :return: 
    """
    base_path = join('.', 'model_data')
    model_path = join(base_path, 'best_model.model')
    data_path = join(base_path, 'results_df.csv')
    torch.save(model.state_dict(), model_path)
    res_df.to_csv(data_path)
    print(f'Results saved in: {base_path}')
    return