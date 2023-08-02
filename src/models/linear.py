import torch 
import torch.nn as nn 
import math 


class Linear(nn.Module): 
    
    def __init__(self, data_shape, target_size): 
        super().__init__() 
        input_size = math.prod(data_shape) 
        self.linear = nn.Linear(input_size, target_size) 
        
    def feature(self, x): 
        x = x.reshape(x.size(0), -1)
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