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

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence



max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

SEED = 407  # used to set random seed



# determining device to use
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# HELPER FUNCTIONS FOR PREPROCESSING
# method to help with pickling
def pickleDataset(dataset_list, path):
    # dataset_list is list of datasets
    # each item will have a field for text & field for label
    # can recreate dataset by pickling fields & examples

    all_dataset_components = []
    for dataset in dataset_list:
        text_field = dataset.fields['text']
        label_field = dataset.fields['label']
        dataset_examples = dataset.examples

        # creating list to store objects
        dataset_components = [text_field, label_field, dataset_examples]
        all_dataset_components.append(dataset_components)

    # pickling list
    with open(path, "wb") as f:
        dill.dump(all_dataset_components, f)

    return


# method to unpickle and create dataset
def unpickleDataset(path):
    # reference pickleDataset method

    # loading pickled dataset components
    with open(path, "rb")as f:
        all_dataset_components = dill.load(f)

    # looping through the multiple datasets
    dataset_list = []
    for dataset_components in all_dataset_components:
        # creating dataset object
        dataset = data.dataset.Dataset(examples=dataset_components[2],
                                       fields={'text': dataset_components[0], 'label': dataset_components[1]})
        dataset_list.append(dataset)

    return dataset_list


"""

# PREPROCESSING

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# splitting test into test and val
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

# padding to ensure appropriate dims
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_val = sequence.pad_sequences(X_val, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
print('X_test shape:', X_test.shape)

# converting np arrays to tensors
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

# creating data loader
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

# pickling
pickle_out = open('imdb_tensors.pkl', 'wb')
pickle.dump([train_loader, val_loader, test_loader], pickle_out)
pickle_out.close()


"""

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
pickleDataset([train_data, val_data, test_data], 'imdb_torchtext_datasets.pkl')

"""

# MODELLING


# below is for bag of words version
# pickle_in = open('imdb_tensors.pkl', 'rb')
# train_loader, val_loader, test_loader = pickle.load(pickle_in)


# below is for word embeddings version
train_data, val_data, test_data = unpickleDataset('imdb_torchtext_datasets.pkl')

# creating iterators
train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    device=device)

train_loader = train_iterator
val_loader = val_iterator


# defining model
class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, lstm_dropout_prob, dropout_prob, n_layers=1):
        super(Net, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=lstm_dropout_prob, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()  # idk if this line is needed
        embeds = self.embedding(x)    # ???
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.fc1(lstm_out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]

        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden



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
# hyperparams = randomSearch(20)

hyperparam_results = []
for params in hyperparams:
    model = Net(vocab_size=max_features, embedding_dim=maxlen, hidden_dim=params['hidden_dim'], lstm_dropout_prob=params['lstm_dropout'], dropout_prob=params['dropout'])
    model.to(device)
    print(model)

    # creating loss & optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())


    epochs = 200
    counter = 0
    print_every = 1000
    clip = 5
    valid_loss_min = np.Inf
    erstop_patience = 10

    val_loss_across_epochs = []


    model.train()
    for i in range(epochs):
        h = model.init_hidden(batch_size)
        for inputs, labels in train_loader:
            counter += 1
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output, h = model(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                """
                for inp, lab in val_loader:
                    val_h = tuple([each.data for each in val_h])
                    inp, lab = inp.to(device), lab.to(device)
                    out, val_h = model(inp, val_h)
                    val_loss = criterion(out.squeeze(), lab.float())
                    val_losses.append(val_loss.item())

                val_loss_mean = np.mean(val_losses)
                """

                model.train()
                print("Epoch: {}/{}...".format(i+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()))

                # print("Epoch: {}/{}...".format(i + 1, epochs),
                #       "Step: {}...".format(counter),
                #       "Loss: {:.6f}...".format(loss.item()),
                #       "Val Loss: {:.6f}".format(val_loss_mean))

        """
        # THIS HAPPENS AT END OF EPOCH -- earlystopping like in keras
        val_h = model.init_hidden(batch_size)
        val_losses = []
        model.eval()
        for inp, lab in val_loader:
            val_h = tuple([each.data for each in val_h])
            inp, lab = inp.to(device), lab.to(device)
            out, val_h = model(inp, val_h)
            val_loss = criterion(out.squeeze(), lab.float())
            val_losses.append(val_loss.item())

        val_loss_mean = np.mean(val_losses)
        val_loss_across_epochs.append(val_loss_mean)
        model.train()

        # checking to see if earlystopping criteria is met
        recent_epochs = val_loss_across_epochs[len(val_loss_across_epochs)-1-erstop_patience:] # patience is number of epochs to wait before stopping

        if min(recent_epochs) == recent_epochs[0]:
            # if the minimum
            break
        """


    # if np.mean(val_losses) <= valid_loss_min:
    #     torch.save(model.state_dict(), './state_dict.pt')
    #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
    #                                                                                     np.mean(val_losses)))
    #     valid_loss_min = np.mean(val_losses)



    """
    test_losses = []
    num_correct = 0
    h = model.init_hidden(batch_size)

    model.eval()
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, h = model(inputs, h)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze())  # Rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc*100))

    # adding results of random search to list
    hyperparam_results.append([test_acc, np.mean(test_losses), params])

    """


    # Test loss: 0.433
    # Test accuracy: 79.904%


# print(hyperparam_results)



