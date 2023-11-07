import torch
from torch import nn,optim

def training(epochs:int,
             dataset:tuple,
             model:nn.Module,
             loss_fn:nn.Module,
             optimizer:optim.Optimizer):
    data = dataset[0]
    targets = dataset[1]
    losses = []
    model.train()
    print('Training ...')
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(data)
        loss = loss_fn(preds,targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: loss: {loss.data.item()}')
    print(f'Epoch {epochs}: loss: {losses[-1]}')
    print('Training complete')
    return losses