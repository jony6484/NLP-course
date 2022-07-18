import argparse
import os
import yaml
from box import Box
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from dataloading import TweetDataset
from modeling import TweetNet
from utils import *
from consts import *


def train(training_args):
    # Setting up logging
    wandb.init(project=PROJECT_NAME, name=training_args.name, config=training_args)

    pprint(training_args)

    # Check args and unpack them
    check_args(training_args)
    data_args, model_args = training_args.data_args, training_args.model_args

    # Set seed for reproducibility
    set_seed(training_args.seed)

    print("Loading datasets")
    train_dataset = TweetDataset(data_args, DATA_DIR + '/' + (TRAIN + CSV))
    train_dataloader = DataLoader(train_dataset, data_args.batch_size, shuffle=data_args.shuffle)
    if training_args.do_eval:
        dev_dataset = TweetDataset(data_args, DATA_DIR + '/' + (DEV + CSV), train_dataset.vocab)
        dev_dataloader = DataLoader(dev_dataset, data_args.eval_batch_size)
    if training_args.do_test:
        test_dataset = TweetDataset(data_args, DATA_DIR + '/' + (TEST + CSV), train_dataset.vocab, training_args.do_test)
        test_dataloader = DataLoader(test_dataset, data_args.eval_batch_size)

    print("Initializing model")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TweetNet(model_args, train_dataset.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(training_args.num_epochs):
        print(f"\n\n-------- Epoch: {epoch} --------\n")
        if training_args.do_train:
            train_loop(train_dataloader, model, loss_fn, optimizer, device, epoch)
        if training_args.do_eval_on_train:
            eval_loop(train_dataloader, model, loss_fn, device, TRAIN, epoch)
        if training_args.do_eval:
            eval_loop(dev_dataloader, model, loss_fn, device, DEV, epoch)

    if training_args.do_test:
        test_loop(test_dataloader, model, device, TEST)

    return


def train_loop(dataloader, model, loss_fn, optimizer, device, epoch):
    model.train()

    for iter_num, (input_ids, lengths, labels) in enumerate(tqdm(dataloader, desc="Train Loop")):
        input_ids, labels = input_ids.to(device), labels.to(device)

        # Compute prediction and loss
        logits = model(input_ids, lengths)
        loss = loss_fn(logits, labels)
        loss = loss / training_args.accumulation_steps

        # Backpropagation
        loss.backward()
        # accumulate gradients: preform a optimization step every training_args.accumulate_grad_batches iterations, or when you reach the end of the epoch
        if ((iter_num + 1) % training_args.accumulation_steps == 0) or (iter_num + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()
        # Remember: iter_num starts at 0. If you set training_args.accumulate_grad_batches to 3, you want to preform your first optimization at the third iteration.


        # Log loss
        wandb.log({"train_loop_loss": loss, EPOCH: epoch, ITERATION: iter_num})


def eval_loop(dataloader, model, loss_fn, device, split, epoch):
    # Change model to eval mode
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    average_loss, correct = 0, 0

    with torch.no_grad():
        for iter_num, (input_ids, lengths, labels) in enumerate(tqdm(dataloader, desc=f"Eval loop on {split}")):
            input_ids, labels = input_ids.to(device), labels.to(device)

            logits = model(input_ids, lengths)

            # Compute metrics
            average_loss += loss_fn(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).float().sum().item()

    # Aggregate metrics
    average_loss /= num_batches
    accuracy = correct / size

    # Log metrics, report everything twice for cross-model comparison too
    wandb.log({f"{split}_average_loss": average_loss, EPOCH: epoch})
    wandb.log({f"{split}_accuracy": accuracy, EPOCH: epoch})


def test_loop(dataloader, model, device, split):
    # Change model to eval mode
    model.eval()

    prediction = []
    with torch.no_grad():
        for iter_num, (input_ids, lengths) in enumerate(tqdm(dataloader, desc=f"Eval loop on {split}")):
            input_ids = input_ids.to(device)
            logits = model(input_ids, lengths)
            prediction += (logits.argmax(dim=1).tolist())
    save_competitive(DATA_DIR + '/' + (TEST + CSV), DATA_DIR + '/' + (COMPETITIVE + CSV), prediction)


def parse_args_for_sweep():
    parser = argparse.ArgumentParser(description='Train an LSTM model on the IMDB dataset.')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to YAML config file. Defualt: config.yaml')
    parser.add_argument('--learning_rate', default=None, type=float,
                        help='learning_rate')
    parser.add_argument('--accumulation_steps', default=None, type=int,
                        help='learning_rate')
    parser.add_argument('--hidden_size', default=None, type=int,
                        help='learning_rate')
    parser.add_argument('--num_layers', default=None, type=int,
                        help='learning_rate')
    args = parser.parse_args()
    with open(args.config) as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))
    for arg, value in args._get_kwargs():
        if value is not None:
            if arg in training_args.keys():
                training_args[arg] = value
            elif arg in training_args.model_args.lstm_args.keys():
                training_args.model_args.lstm_args[arg] = value
    return training_args


if __name__ == '__main__':
    training_args = parse_args_for_sweep()
    train(training_args)


