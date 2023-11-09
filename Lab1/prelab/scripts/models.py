import torch
from torch import nn

class myRNN(nn.Module):
    def __init__(self,input_shape,hidden_size,output_size):
        super().__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=self.input_shape, hidden_size=self.hidden_size)
        self.linear = nn.Linear(self.hidden_size,self.output_size)
        
    def forward(self,x):
        h0 = torch.randn(1,self.hidden_size)
        out, _ = self.rnn(x,h0)
        ret = self.linear(out)
        return ret
    
class myLSTM(nn.Module):
    def __init__(self,input_shape,hidden_size,output_size):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(self.input_shape,hidden_size=self.hidden_size)
        self.linear = nn.Linear(self.hidden_size,self.output_size)
    
    def forward(self,x):
        h0 = torch.randn(1,self.hidden_size)
        c0 = torch.randn(1,self.hidden_size)
        out, _ = self.lstm(x,(h0,c0))
        ret = self.linear(out)
        return ret
    
class myGRU(nn.Module):
    def __init__(self,input_shape,hidden_size,output_size):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size=self.input_shape, hidden_size=self.hidden_size)
        self.linear = nn.Linear(self.hidden_size,self.output_size)
        
    def forward(self,x):
        h0 = torch.randn(1,self.hidden_size)
        out, _ = self.gru(x,h0)
        ret = self.linear(out)
        return ret
        