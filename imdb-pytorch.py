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
from torch.utils.data import TensorDataset, DataLoader
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

# creating iterators
# train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
#     (train_data, val_data, test_data),
#     batch_size=batch_size,
#     sort_within_batch=True,
#     device=device)

train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size=batch_size,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device)

# taken from tutorial
# defining model
class AdamNetV2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers,
                 is_bidirectional=True, dropout=0.0, output_dim=1, padding_idx=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=padding_idx)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=is_bidirectional, dropout=dropout)

        self.fc = nn.Linear((is_bidirectional + 1) * hidden_dim, output_dim)

        self.is_bidirectional = is_bidirectional

    def forward(self, input_sequence, sequence_length):

        embeddings = self.embedding(input_sequence)

        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, sequence_length)

        packed_output, (hidden_state, cell_state) = self.lstm(packed_embeddings)

        if self.is_bidirectional:
            output = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        else:
            output = hidden_state[-1, :, :]


        scores = self.fc(output)

        return scores




# randomly selects num_combos # of params
def randomSearch(num_combos):
    params = []
    while (len(params) < num_combos):
        # params for LSTM
        lstm_dropout = random.uniform(0, .5)
        dropout = random.uniform(0, .5)
        hidden_dim = random.randint(128, 1024)

        value = {'lstm_dropout': lstm_dropout, 'dropout': dropout, 'hidden_dim': hidden_dim}
        params.append(value)

    return params





hyperparams = [{'lstm_dropout': .2, 'dropout': .2, 'hidden_dim': 512}]
hyperparams = randomSearch(20)

vocab_size = len(TEXT.vocab)
embedding_dim = 300  # This needs to match the size of the pre-trained embeddings!
hidden_dim = 256
num_layers = 3
dropout = 0.5
pad_idx = TEXT.vocab.stoi[TEXT.pad_token]


model = AdamNetV2(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                  n_layers=num_layers,  dropout=dropout, padding_idx=pad_idx)

# Initialize word embeddings
glove_vectors = TEXT.vocab.vectors
model.embedding.weight.data.copy_(glove_vectors)

# Zero out <unk> and <pad> tokens
unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

# Define our loss function, optimizer, and move things to GPU
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters())


def accuracy(scores, y):
    scores = torch.round(torch.sigmoid(scores))
    correct = (scores == y)
    acc = int(correct.sum()) / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.text

        predictions = model(text, text_lengths).squeeze(1)

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

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



num_epochs = 10
best_valid_loss = 1000000

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, val_iterator, criterion)

    # Print test performance
    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f}\nTest Acc: {test_acc*100:.2f}%')




# hyperparam_results = []
# for params in hyperparams:
#     model = Net(vocab_size=max_features, embedding_dim=embedding_dim, hidden_dim=params['hidden_dim'], lstm_dropout_prob=params['lstm_dropout'], dropout_prob=params['dropout'])
#     model.to(device)
#     print(model)
#
#     # creating loss & optimizer
#     criterion = nn.BCELoss()
#     optimizer = torch.optim.Adam(model.parameters())
#
#
#     epochs = 200
#     counter = 0
#     print_every = 1000
#     clip = 5
#     valid_loss_min = np.Inf
#     erstop_patience = 10
#
#     val_loss_across_epochs = []
#
#
#     model.train()
#     for i in range(epochs):
#         h = model.init_hidden(batch_size)
#         for inputs, labels in train_loader:
#             counter += 1
#             h = tuple([e.data for e in h])
#             inputs, labels = inputs.to(device), labels.to(device)
#             model.zero_grad()
#             output, h = model(inputs, h)
#             loss = criterion(output.squeeze(), labels.float())
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), clip)
#             optimizer.step()
#
#             if counter % print_every == 0:
#                 val_h = model.init_hidden(batch_size)
#                 val_losses = []
#                 model.eval()
#                 """
#                 for inp, lab in val_loader:
#                     val_h = tuple([each.data for each in val_h])
#                     inp, lab = inp.to(device), lab.to(device)
#                     out, val_h = model(inp, val_h)
#                     val_loss = criterion(out.squeeze(), lab.float())
#                     val_losses.append(val_loss.item())
#
#                 val_loss_mean = np.mean(val_losses)
#                 """
#
#                 model.train()
#                 print("Epoch: {}/{}...".format(i+1, epochs),
#                       "Step: {}...".format(counter),
#                       "Loss: {:.6f}...".format(loss.item()))
#
#                 # print("Epoch: {}/{}...".format(i + 1, epochs),
#                 #       "Step: {}...".format(counter),
#                 #       "Loss: {:.6f}...".format(loss.item()),
#                 #       "Val Loss: {:.6f}".format(val_loss_mean))
#
#         """
#         # THIS HAPPENS AT END OF EPOCH -- earlystopping like in keras
#         val_h = model.init_hidden(batch_size)
#         val_losses = []
#         model.eval()
#         for inp, lab in val_loader:
#             val_h = tuple([each.data for each in val_h])
#             inp, lab = inp.to(device), lab.to(device)
#             out, val_h = model(inp, val_h)
#             val_loss = criterion(out.squeeze(), lab.float())
#             val_losses.append(val_loss.item())
#
#         val_loss_mean = np.mean(val_losses)
#         val_loss_across_epochs.append(val_loss_mean)
#         model.train()
#
#         # checking to see if earlystopping criteria is met
#         recent_epochs = val_loss_across_epochs[len(val_loss_across_epochs)-1-erstop_patience:] # patience is number of epochs to wait before stopping
#
#         if min(recent_epochs) == recent_epochs[0]:
#             # if the minimum
#             break
#         """
#
#
#     # if np.mean(val_losses) <= valid_loss_min:
#     #     torch.save(model.state_dict(), './state_dict.pt')
#     #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
#     #                                                                                     np.mean(val_losses)))
#     #     valid_loss_min = np.mean(val_losses)
#
#
#
#     """
#     test_losses = []
#     num_correct = 0
#     h = model.init_hidden(batch_size)
#
#     model.eval()
#     for inputs, labels in test_loader:
#         h = tuple([each.data for each in h])
#         inputs, labels = inputs.to(device), labels.to(device)
#         output, h = model(inputs, h)
#         test_loss = criterion(output.squeeze(), labels.float())
#         test_losses.append(test_loss.item())
#         pred = torch.round(output.squeeze())  # Rounds the output to 0/1
#         correct_tensor = pred.eq(labels.float().view_as(pred))
#         correct = np.squeeze(correct_tensor.cpu().numpy())
#         num_correct += np.sum(correct)
#
#     print("Test loss: {:.3f}".format(np.mean(test_losses)))
#     test_acc = num_correct/len(test_loader.dataset)
#     print("Test accuracy: {:.3f}%".format(test_acc*100))
#
#     # adding results of random search to list
#     hyperparam_results.append([test_acc, np.mean(test_losses), params])
#
#     """
#
#
#     # Test loss: 0.433
#     # Test accuracy: 79.904%
#
#
# # print(hyperparam_results)
#
#
#
