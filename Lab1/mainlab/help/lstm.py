import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

import subprocess
from help.parser import parser, make_scale_fn, train_test_split
#from help.lstm import BasicLSTM
import numpy as np

import matplotlib.pyplot as plt
from help.plot_confusion_matrix import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
from help.parser import parser
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



patience=3
epochs=15

class EarlyStopping(object):
    def __init__(self,model,save_dir, patience, mode="min", base=None):
        self.best = base
        self.patience = patience
        self.patience_left = patience
        self.mode = mode
        self.model = model
        self.save_dir = save_dir

    def stop(self, value: float) -> bool:
        # TODO: YOUR CODE HERE
        # Decrease patience if the metric hs not improved
        # Stop when patience reaches zero
        
        ## checkpoint and early stopping implementation
        if self.best == None:    ## first epoch
            self.best = value 
            torch.save((self.model).state_dict(),self.save_dir)
            
        elif self.best < value: ## if error increases, decreace patience
            self.patience_left -= 1
        elif self.best > value: ## if error decreases, restart patience with new error, resave best params
            self.best = value
            self.patience_left = self.patience
            torch.save((self.model).state_dict(),self.save_dir)
            
        return self.patience_left == 0

    def has_improved(self, value: float) -> bool:
        # TODO: YOUR CODE HERE
        # Check if the metric has improved
        return self.best < value

class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
        feats: Python list of numpy arrays that contain the sequence features.
               Each element of this list is a numpy array of shape seq_length x feature_dimension
        labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        # TODO: YOUR CODE HERE
        self.lengths = [len(i) for i in feats]

        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype("int64")

    def zero_pad_and_stack(self, x: np.ndarray) -> np.ndarray:
        """
        This function performs zero padding on a list of features and forms them into a numpy 3D array
        returns
            padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #
        max_len = max(self.lengths) ## max sequence length
        ## stack zeros np.array of shape (maxlen - seqlen x feature len) at the end of each seq array (axis 0)
        padded = np.array([np.concatenate((seq,np.zeros((max_len-seq.shape[0],seq.shape[1]))),axis=0) for seq in x],dtype=np.float32)  
        
        return padded

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=False,
        dropout=0.0,
    ):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size*2 if self.bidirectional else rnn_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        
        ## the lstm part
        self.lstm = nn.LSTM(input_size=self.input_dim,hidden_size=self.feature_size,
                            num_layers=self.num_layers,bidirectional=self.bidirectional,
                            dropout=self.dropout,batch_first=True)
        
        ## the  output part
        self.hidden2out_dim = 2*self.feature_size if self.bidirectional else self.feature_size
        self.output = nn.Linear(in_features=self.hidden2out_dim,out_features=self.output_dim)
        

    def forward(self, x, lengths):
        """
        x : 3D numpy array of dimension N x L x D
            N: batch index
            L: sequence index
            D: feature index

        lengths: N x 1
        """
        # --------------- Insert your code here ---------------- #

        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network

        dim = 2*self.num_layers if self.bidirectional else self.num_layers ## determines 1st dim of h0,c0
        
        h0 = torch.zeros(dim,x.size(0),self.feature_size) ## batched input
        c0 = torch.zeros(dim,x.size(0),self.feature_size)
        out, _ = self.lstm(x,(h0,c0))
        last_steps = self.last_timestep(outputs=out,lengths=lengths,bidirectional=self.bidirectional)
        
        last_outputs = self.output(last_steps)
        return last_outputs ## size: (N,output dim)

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
        Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs) ## get the seperate vectors
            last_forward = self.last_by_index(forward, lengths) ## for forward use last_by_index to get last
            last_backward = backward[:, 0, :] ## in backwards the first is the last
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        '''
        Splits outputs vectors to forward and backward part
        '''
        direction_size = int(outputs.size(-1) / 2) ## forward and backward have the half len of outputs
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        '''
        Extract the last relevant output from a sequnence of ouputs using 
        the provided lenghts.
        
        outputs: (N,L,D) = (batch, sequence, feature) sizes.
        lenghts: Nx1
        
        '''
        # Index of the last output for each sequence.
        idx = (
            (lengths - 1) ## get the index of the last item for each sequence in outputs (turn lenghts from values to indexes of said values)
            .view(-1, 1)  ## reshape tensor into col vector
            .expand(outputs.size(0), outputs.size(2)) ## broadcast the vector to batch dim (0) and feature dim (2) -> create matrix of number=feature size vectors (side by side)
            .unsqueeze(1) ## add a new dimension along the 2nd axis to generate indexes of proper size, to use them on outputs (2D to 3D) -> size (batch x 1 x feature size)
        )
        
        ## get the elements from outputs using the idx 3D matrix, along the 2nd dim(=1) (the lenghts dimension)
        return outputs.gather(1, idx).squeeze()


def create_dataloaders(batch_size):
    X, X_test, y, y_test, spk, spk_test = parser("./recordings", n_mfcc=13)

    X_train, X_val, y_train, y_val, spk_train, spk_val = train_test_split(
        X, y, spk, test_size=0.2, stratify=y
    )

    trainset = FrameLevelDataset(X_train, y_train)
    validset = FrameLevelDataset(X_val, y_val)
    testset = FrameLevelDataset(X_test, y_test)
    
    # Initialize the training, val and test dataloaders (torch.utils.data.DataLoader)
    train_dataloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
    val_dataloader = DataLoader(validset,batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(testset,batch_size=batch_size,shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def training_loop(model, train_dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for num_batch, batch in enumerate(train_dataloader):
        features, labels, lengths = batch
        # TODO: YOUR CODE HERE
        # zero grads in the optimizer
        optimizer.zero_grad()
        # run forward pass
        predictions = model(features,lengths)
        # calculate loss
        loss = criterion(predictions,labels)
        # TODO: YOUR CODE HERE
        # Run backward pass
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        num_batches += 1
    train_loss = running_loss / num_batches
    return train_loss


def evaluation_loop(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    y_pred = torch.empty(0, dtype=torch.int64)
    y_true = torch.empty(0, dtype=torch.int64)
    with torch.no_grad():
        for num_batch, batch in enumerate(dataloader):
            features, labels, lengths = batch

            # TODO: YOUR CODE HERE
            # Run forward pass
            logits = model(features,lengths)
            # calculate loss
            loss = criterion(logits,labels)
            running_loss += loss.data.item()
            # Predict
            outputs = torch.argmax(logits,dim=1) # Calculate the argmax of logits
            y_pred = torch.cat((y_pred, outputs))
            y_true = torch.cat((y_true, labels))
            num_batches += 1
    valid_loss = running_loss / num_batches
    return valid_loss, y_pred, y_true


def train(bidirect,dropout,weight_decay,train_dataloader, val_dataloader, criterion,save_dir):
    # TODO: YOUR CODE HERE
    input_dim = train_dataloader.dataset.feats.shape[-1]
    model = BasicLSTM(
        input_dim,
        rnn_size=64,
        output_dim=10,
        num_layers=2,
        bidirectional=bidirect,
        dropout=dropout,
    )
    # TODO: YOUR CODE HERE
    # Initialize AdamW
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=1e-2,weight_decay=weight_decay)

    ## lists to save each epochs losses for plotting
    train_losses = {}
    valid_losses = {}
    early_stopping = EarlyStopping(model,save_dir,patience, mode="min")
    
    print(f'Training for dropout:{dropout} and weight decay:{weight_decay}')
    for epoch in range(epochs):
        training_loss = training_loop(model, train_dataloader, optimizer, criterion)
        valid_loss, y_pred, y_true = evaluation_loop(model, val_dataloader, criterion)
        # TODO: Calculate and print accuracy score
        valid_accuracy = accuracy_score(y_pred=y_pred,y_true=y_true) ### TODO: calculate accuraccy fro y_pred and y_true
        print(
            "Epoch {}: train loss = {}, valid loss = {}, valid acc = {}".format(
                epoch, training_loss, valid_loss, valid_accuracy
            )
        )
        train_losses[epoch]=training_loss
        valid_losses[epoch]=valid_loss
        
        if early_stopping.stop(valid_loss):
            print("early stopping...")
            break

    return model, train_losses, valid_losses





##### DROPOUT = 0.0 WEIGHT DECAY = 0.0 ###
output_dim = 10  # number of digits
# TODO: YOUR CODE HERE
# Play with variations of these hyper-parameters and report results
rnn_size = 64
num_layers = 2
bidirectional = False
dropout = 0.0
batch_size = 32
patience = 3
epochs = 15
lr = 1e-3
weight_decay = 0.0
checkpoint_loc = f'./models_dict/dr_{dropout}_wd_{weight_decay}.pth'  ### checkpoints save location

train_dataloader, val_dataloader, test_dataloader = create_dataloaders(batch_size)
# TODO: YOUR CODE HERE
# Choose an appropriate loss function
criterion = nn.CrossEntropyLoss()

model,train_losses,valid_losses = train(bidirectional,dropout,weight_decay,train_dataloader, val_dataloader, criterion, checkpoint_loc)

## plot results
plt.plot(np.linspace(0,len(train_losses),len(train_losses)),train_losses.values())
plt.plot(np.linspace(0,len(train_losses),len(train_losses)),valid_losses.values())
plt.legend(['Train loss','Valid loss'])
plt.xlabel('Epochs')
plt.title(f'Dropout={dropout},weight decay={weight_decay}')
plt.show()

## using evaluation loop on test data
test_loss, test_pred, test_true = evaluation_loop(model, test_dataloader, criterion)
# TODO: YOUR CODE HERE
# print test loss and test accuracy
print(f'Test loss:{test_loss}, test acc: {accuracy_score(test_pred,test_true)}')



#### DROPOUT=0.4 WEIGHT DECAY=0.1 #####

rnn_size = 64
num_layers = 2
bidirectional = False
dropout = 0.4
batch_size = 32
patience = 3
epochs = 15
lr = 1e-3
weight_decay = 0.1
checkpoint_loc = f'./models_dict/dr_{dropout}_wd_{weight_decay}.pth'  ### checkpoints save location

# TODO: YOUR CODE HERE
# Choose an appropriate loss function
criterion = nn.CrossEntropyLoss()
model,train_losses,valid_losses = train(bidirectional,dropout,weight_decay,train_dataloader, val_dataloader, criterion, checkpoint_loc)

## plot results
plt.plot(np.linspace(0,len(train_losses),len(train_losses)),train_losses.values())
plt.plot(np.linspace(0,len(train_losses),len(train_losses)),valid_losses.values())
plt.legend(['Train loss','Valid loss'])
plt.xlabel('Epochs')
plt.title(f'Dropout={dropout},weight decay={weight_decay}')
plt.show()

## using evaluation loop on test data
test_loss, test_pred, test_true = evaluation_loop(model, test_dataloader, criterion)
# TODO: YOUR CODE HERE
# print test loss and test accuracy
print(f'Test loss:{test_loss}, test acc: {accuracy_score(test_pred,test_true)}')



#### DROPOUT = 0.0 WEIGHT DEACY=0.1 BIDIRECTIONAL ####


rnn_size = 64
num_layers = 2
bidirectional = True
dropout = 0.4
batch_size = 32
patience = 3
epochs = 15
lr = 1e-3
weight_decay = 0.1
checkpoint_loc = f'./models_dict/dr_{dropout}_wd_{weight_decay}_bidirectional.pth'  ### checkpoints save location

# TODO: YOUR CODE HERE
# Choose an appropriate loss function
criterion = nn.CrossEntropyLoss()
print('Bidirectional LSTM')
model,train_losses,valid_losses = train(bidirectional,dropout,weight_decay,train_dataloader, val_dataloader, criterion, checkpoint_loc)

## plot results
plt.plot(np.linspace(0,len(train_losses),len(train_losses)),train_losses.values())
plt.plot(np.linspace(0,len(train_losses),len(train_losses)),valid_losses.values())
plt.legend(['Train loss','Valid loss'])
plt.xlabel('Epochs')
plt.title(f'Dropout={dropout},weight decay={weight_decay},bidirectional')
plt.show()

## using evaluation loop on test data
test_loss, test_pred, test_true = evaluation_loop(model, test_dataloader, criterion)
# TODO: YOUR CODE HERE
# print test loss and test accuracy
print(f'Test loss:{test_loss}, test acc: {accuracy_score(test_pred,test_true)}')







#### BEST MODEL so far: bidirectional, dropout = 0.4, weight decay>0
input_dim = train_dataloader.dataset.feats.shape[-1]

best_model = BasicLSTM(input_dim,rnn_size=64,output_dim=10,num_layers=2,bidirectional=True,dropout=0.4)

best_model.load_state_dict(torch.load('./models_dict/dr_0.4_wd_0.1_bidirectional.pth'))
best_model.eval()

## Get test and vali loss, and their predictions
test_loss, test_pred, test_true = evaluation_loop(best_model, test_dataloader, criterion)
valid_loss, y_pred, y_true = evaluation_loop(best_model, val_dataloader, criterion)

print('Best model results:')
print(f'Test loss:{test_loss}, test acc: {accuracy_score(test_pred,test_true)}')
print(f'Valid loss:{valid_loss}, test acc: {accuracy_score(y_pred,y_true)}')


valid_cm = confusion_matrix(y_true,y_pred)
test_cm = confusion_matrix(test_true,test_pred)
classes = [i for i in range(10)]
print('For test results:')
plot_confusion_matrix(test_cm,classes)
plt.show()
print('For valid results:')
plot_confusion_matrix(valid_cm,classes)
plt.show()