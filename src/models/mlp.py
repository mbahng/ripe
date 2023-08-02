import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, data_shape, hidden_size, scale_factor, num_layers, activation, target_size):
        super().__init__()
        input_size = math.prod(data_shape)
        blocks = []
        for _ in range(num_layers):
            blocks.append(nn.Linear(input_size, hidden_size))
            if activation == 'relu':
                blocks.append(nn.ReLU())
            elif activation == 'sigmoid':
                blocks.append(nn.Sigmoid())
            else:
                raise ValueError('Not valid activation')
            input_size = hidden_size
            hidden_size = int(hidden_size * scale_factor)
        self.blocks = nn.Sequential(*blocks)
        self.linear = nn.Linear(input_size, target_size)
        
        
        parameter_list = [W for W in self.parameters()]
        
    def feature(self, x): 
        x = x.reshape(x.size(0), -1)
        x = self.blocks(x) 
        return x 
    
    def classify(self, x): 
        x = self.linear(x) 
        return x 
    
    def f(self, x): 
        x = self.feature(x)
        x = self.classify(x) 
        return x 
    
    def forward(self, x): 
        return self.f(x)


class dropoutMLP(nn.Module): 
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 30)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.dropout(x)
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x