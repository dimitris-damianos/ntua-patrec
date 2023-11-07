import torch
from torch import nn

class myRNN(nn.Module):
    def __init__(self,input_shape,hidden_size=10):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=self.input_shape, hidden_size=self.hidden_size)
        
    def forward(self,x):
        h0 = torch.randn(1,self.hidden_size)
        out, _ = self.rnn(x,h0)
        return out
    
class myLSTM(nn.Module):
    def __init__(self,input_shape,hidden_size=10):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.input_shape,hidden_size=self.hidden_size)
    
    def forward(self,x):
        h0 = torch.randn(1,self.hidden_size)
        c0 = torch.randn(1,self.hidden_size)
        out, _ = self.lstm(x,(h0,c0))
        return out
    
class myGRU(nn.Module):
    def __init__(self,input_shape,hidden_size=10):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=self.input_shape, hidden_size=self.hidden_size)
        
    def forward(self,x):
        h0 = torch.randn(1,self.hidden_size)
        out, _ = self.gru(x,h0)
        return out