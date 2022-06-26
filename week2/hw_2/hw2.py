import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
import gensim
import gensim.downloader
from gensim.models import KeyedVectors
from sklearn.preprocessing import OneHotEncoder
matplotlib.style.use('bmh')
from os.path import exists


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
        self.X = np.zeros(shape=(len(self.reviews), self.tokenizer.vector_size))
        N = len(self.reviews)
        for ii, review in enumerate(self.reviews):
            print(f'tokenizing: {ii+1}/{N}|{int(100*ii/N)*"="}{int(100*(N-ii)/N)*"-"}|', end='\r')
            self.X[ii, :] = self.make_review_token(review)
        print("")
        self.X = torch.tensor(self.X, dtype=torch.float32)

    def make_review_token(self, review):
        review = [word.strip('.,;!@#$%^&*()/"\'<>~') for word in review.lower().split() if len(word) > 1]
        tokens = []
        for word in review:
            if word not in self.tokenizer.key_to_index:
                continue
            tokens.append(self.tokenizer[word])
        feat_array = np.concatenate(tokens, axis=0).reshape(-1, self.tokenizer.vector_size)
        X_i = feat_array.mean(0)
        # X_i = np.median(feat_array, axis=0)
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
        X = self.hidden_layer1(X)
        X = F.relu(X)
        X = self.hidden_layer2(X)
        X = F.relu(X)
        X = self.output_layer(X)
        return X


def make_model(input_dim, num_classes, hidden_dim):
    model = GloveClassifier(input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            num_classes=num_classes)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    return model, loss_fun, optimizer


def train_model(train_loader, test_loader, model, loss_fun, optimizer, device, num_epochs=2):
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
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

            batch_loss.append(loss.item())
            batch_acc.append(100*torch.mean((torch.argmax(logits, axis=1) == y).float().cpu()).item())
            print(f'batch:{ii + 1}/{Nb}|{int(100 * ii / Nb) * "="}{int(100 * (Nb - ii) / Nb) * "-"}'
                  f'|loss:{batch_loss[ii]:0.3f}|accuracy:{batch_acc[ii]:0.2f}%', end='\r')

        train_loss.append(np.mean(batch_loss))
        train_acc.append((np.mean(batch_acc)))
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                X = batch['input_vectors'].to(device)
                y = batch['labels'].to(device)
                logits = model(X)
                loss = loss_fun(logits, y)
                batch_loss.append(loss.item())
                batch_acc.append(100*torch.mean((torch.argmax(logits, axis=1) == y).float().cpu()).item())
            test_loss.append(np.mean(batch_loss))
            test_acc.append((np.mean(batch_acc)))
        print(f'epoch:{epoch_i + 1}/{num_epochs}|loss Tr/Ts: {train_loss[epoch_i]:0.3f}/{test_loss[epoch_i]:0.3f}|'
              f'accuracy Tr/Ts: {train_acc[epoch_i]:0.2f}/{test_acc[epoch_i]:0.2f}%')
    model.to('cpu')
    return train_loss, test_loss, train_acc, test_acc, model


def main():
    data_train = pd.read_csv('IMDB_train.csv')
    data_test = pd.read_csv('IMDB_test.csv')
    token_model = 'glove-twitter-200'
    if exists(token_model + ".model"):
        tokenizer = KeyedVectors.load(token_model + ".model")
    else:
        tokenizer = gensim.downloader.load(token_model)
        tokenizer.save(token_model + ".model")
    dataset_train = ReviewDataset(data_train, tokenizer)
    dataset_test = ReviewDataset(data_test, tokenizer, dataset_train.label_2_idx)
    print('done loading')
    batch_size = 128
    train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)#, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(dataset_test, shuffle=False, batch_size=2*batch_size)#, num_workers=2, persistent_workers=True)
    model, loss_fun, optimizer = make_model(input_dim=dataset_train.tokenizer.vector_size, hidden_dim=512,
                                            num_classes=len(dataset_train.label_2_idx))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss, test_loss, train_acc, test_acc, model = train_model(
        train_loader, test_loader, model, loss_fun, optimizer, device, num_epochs=100)
    return


if __name__ == '__main__':
    main()