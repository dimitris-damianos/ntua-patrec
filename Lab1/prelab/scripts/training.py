import torch
from torch import nn,optim
from tqdm import tqdm

def training(epochs:int,
             dataset:tuple,
             model:nn.Module,
             loss_fn:nn.Module,
             optimizer:optim.Optimizer):
    data = dataset[0]
    targets = dataset[1]
    losses = []
    model.train()
    for epoch in tqdm(range(epochs),desc='Training model...'):
        optimizer.zero_grad()
        preds = model(data)
        loss = loss_fn(preds,targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())
    return losses