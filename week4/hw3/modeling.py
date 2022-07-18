import torch
import torch.nn as nn
import numpy as np

from gensim import downloader


class TweetNet(nn.Module):
    def __init__(self, model_args, vocab_size):
        super(TweetNet, self).__init__()
        self.embedding_model = model_args.embedding_model
        self.lstm_args = model_args.lstm_args
        self.hidden_size = self.lstm_args.hidden_size if not self.lstm_args.bidirectional else self.lstm_args.hidden_size * 2
        self.output_size = model_args.output_size
        self.dropout = model_args.dropout

        # Embedding of dim vocab_size x model_args.lstm_args.input_size
        self.embedding = self.load_pretrained_embedding()
        # LSTM
        self.lstm = nn.LSTM(**self.lstm_args)
        # Classifier containing dropout, linear layer and sigmoid
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )

    def forward(self, input_ids, lengths):
        # Embed
        embeds = self.embedding(input_ids)   # (1, seq_length) -> (1, seq_length, input_size)

        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths=lengths, batch_first=True,
                                                                enforce_sorted=False)
        # Run through LSTM and take the final layer's output
          # (1, seq_length, input_size) -> (1, max_seq_length, hidden_size)
        lstm_packed_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed_out, batch_first=True)

        # Take the mean of all the output vectors
        seq_embeddings = (lstm_out.sum(dim=1).t() / lengths.to(lstm_out.device)).t()  # (1, max_seq_length, hidden_size) -> (1, hidden_size)

        # Classifier
        logits = self.classifier(seq_embeddings) # (1, hidden_size) -> (1, n_classes)
        logits = logits.float()
        return logits


    def load_pretrained_embedding(self):
        model = downloader.load(self.embedding_model)
        model.add_vector(len(model), np.zeros(shape=model.get_vector(0).shape))  # Unk vector
        model.add_vector(len(model) + 1, np.zeros(shape=model.get_vector(0).shape)) # Pad vector
        weights = torch.FloatTensor(model.vectors)
        embedding = nn.Embedding.from_pretrained(weights)
        return embedding