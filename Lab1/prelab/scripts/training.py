import torch
from torch import nn,optim
from tqdm import tqdm

### training function 
def training(epochs,
             dataset,
             model,
             loss_fn,
             optimizer):
    data = dataset[0]
    targets = dataset[1]
    losses = []
    seq_len = data.shape[0]
    
    model.train()
    for epoch in tqdm(range(epochs),desc='Training...'):
        optimizer.zero_grad()
        loss = 0
        for X,Y in zip(data,targets):
            preds = model(X.unsqueeze(1))
            loss += loss_fn(preds,Y)
        losses.append(loss.data.item()/seq_len)
        loss.backward()
        optimizer.step()
    return losses