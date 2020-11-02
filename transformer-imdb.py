import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
import pickle
import random
import dill

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext import data
from torchtext import datasets

# constants
batch_size = 32

# pickling path
path_string = 'imdb_torchtext_datasets'

SEED = 407  # used to set random seed



# determining device to use
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# HELPER FUNCTIONS FOR PREPROCESSING -- torchtext doesn't like pickle :/
# method to help with pickling
def pickleDataset(dataset_list, corpus_fields, path):
    # dataset_list is list of datasets
    # each item will have a field for text & field for label
    # can recreate dataset by pickling fields & examples
    # corpus fields are TEXT & LABEL

    all_dataset_components = []
    for dataset in dataset_list:
        text_field = dataset.fields['text']
        label_field = dataset.fields['label']
        dataset_examples = dataset.examples

        # creating list to store objects
        dataset_components = [text_field, label_field, dataset_examples]
        all_dataset_components.append(dataset_components)


    # pickling dataset components
    with open(path+'.pkl', "wb") as f:
        dill.dump(all_dataset_components, f)

    # pickling dataset components
    with open(path+'_corpus.pkl', "wb") as f:
        dill.dump(corpus_fields, f)

    return


# method to unpickle and create dataset
def unpickleDataset(path):
    # reference pickleDataset method

    # loading pickled dataset components
    with open(path+'.pkl', "rb")as f:
        all_dataset_components = dill.load(f)

    # looping through the multiple datasets
    dataset_list = []
    for dataset_components in all_dataset_components:
        # creating dataset object
        dataset = data.dataset.Dataset(examples=dataset_components[2],
                                       fields={'text': dataset_components[0], 'label': dataset_components[1]})
        dataset_list.append(dataset)

    # loading corpus components
    with open(path+'_corpus.pkl', "rb")as f:
        corpus_fields = dill.load(f)

    return dataset_list, corpus_fields


"""
# SECOND METHOD OF PREPROCESSING (WORD EMBEDDINGS)

# reference https://medium.com/@adam.wearne/lets-get-sentimental-with-pytorch-dcdd9e1ea4c9 for tutorial
TEXT = data.Field(lower=True, include_lengths=True)   # tokenize='spacy'??
LABEL = data.LabelField(dtype=torch.float)  # storing labels as float

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, val_data = train_data.split(random_state=random.seed(SEED))

TEXT.build_vocab(train_data,
                 max_size=max_features,
                 vectors='glove.6B.300d',
                 unk_init=torch.Tensor.normal_)

LABEL.build_vocab(train_data)





# pickling using methods
pickleDataset([train_data, val_data, test_data], [TEXT, LABEL], path_string)

"""

# MODELLING

(train_data, val_data, test_data), (TEXT, LABEL) = unpickleDataset(path_string)


train_iterator, val_iterator, test_iterator = data.Iterator.splits(
    (train_data, val_data, test_data),
    batch_size=batch_size,
    device=device)



class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads   # k is embedding dimension

        # These compute the queries, keys and values for all heads (as a single concatenated vector)
        self.to_keys = nn.Linear(k, k * heads, bias=False)
        self.to_queries = nn.Linear(k, k * heads, bias=False)
        self.to_values = nn.Linear(k, k * heads, bias=False)

        # This unifies the outputs of the different heads into a single k-vector
        self.unify_heads = nn.Linear(heads * k, k)

    def forward(self, x):
        # b is batch -- sequence of t vectors of length k --- (b, t, k)
        b, t, k = x.size()
        h = self.heads

        queries = self.to_keys(x).view(b, t, h, k)
        keys = self.to_queries(x).view(b, t, h, k)
        values = self.to_values(x).view(b, t, h, k)

        # fold heads into the batch dimension
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        # scaling
        queries = queries / (k ** (1 / 4))
        keys = keys / (k ** (1 / 4))

        # dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # - dot has size (b*h, t, t) containing raw weights
        dot = F.softmax(dot, dim=2)
        # - dot now contains row-wise normalized weights

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, k)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unify_heads(out)



class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)



class Transformer(nn.Module):
    def __init__(self, k, heads, depth, num_tokens, num_classes, seq_length):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        # The sequence of transformer blocks that does all the
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        # generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        # generate position embeddings
        positions = torch.arange(t, device=device)

        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.tblocks(x)

        # Average-pool over the t dimension and project to class probabilities
        x = self.toprobs(x.mean(dim=1))

        return F.log_softmax(x, dim=1)


def accuracy(scores, y):
    scores = torch.round(torch.sigmoid(scores))
    correct = (scores == y)
    acc = int(correct.sum()) / len(correct)
    return acc

temp = []
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.text
        temp.append(batch)

        # transposing to get correct dims
        text = torch.transpose(text, 0, 1)

        predictions = model(text).squeeze(1)
        temp.append(predictions)

        loss = criterion(predictions, batch.label)
        acc = accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc


    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            # transposing to get correct dims
            text = torch.transpose(text, 0, 1)

            predictions = model(text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



vocab_size = len(TEXT.vocab)
embedding_dim = 300  # This needs to match the size of the pre-trained embeddings!
hidden_dim = 256
num_layers = 3
dropout = 0.5
pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
seq_len = 956


model = Transformer(k=embedding_dim, seq_length=seq_len, heads=1, depth=6, num_tokens=vocab_size, num_classes=1)

    # vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
    #               n_layers=num_layers,  dropout=dropout, padding_idx=pad_idx)


# Initialize word embeddings
glove_vectors = TEXT.vocab.vectors
model.token_emb.weight.data.copy_(glove_vectors)

# Zero out <unk> and <pad> tokens
unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
model.token_emb.weight.data[unk_idx] = torch.zeros(embedding_dim)
model.token_emb.weight.data[pad_idx] = torch.zeros(embedding_dim)


# Define our loss function, optimizer, and move things to GPU
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters())


num_epochs = 10
best_valid_loss = 1000000

print('Entered training loop')
for epoch in range(num_epochs):
    print(epoch)
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    print(train_loss)
    print(train_acc)
    print()
    # valid_loss, valid_acc = evaluate(model, val_iterator, criterion)

    # Print test performance
    # test_loss, test_acc = evaluate(model, test_iterator, criterion)

    # print(f'Test Loss: {test_loss:.3f}\nTest Acc: {test_acc*100:.2f}%')




