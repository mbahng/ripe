import torch
import numpy as np
from src.regularizers import * 
from src.models.linear import Linear
import matplotlib.pyplot as plt

def avgParam(model): 
    out = []
    for param in model.parameters(): 
        out.append(torch.mean(param).item())
    return np.mean(out)

def L0_sparsity(model): 
    total = 0 
    zeros = 0 
    for W in model.parameters(): 
        size = math.prod(W.size())
        total += size
        zeros += size - torch.count_nonzero(W).item()
        
    return zeros/total 

def parameterDistribution(model): 
    vals = []
    for W in model.parameters(): 
        vals += list(W.cpu().data.detach().numpy().reshape(-1))
    plt.hist(vals, bins=np.linspace(-0.1, 0.1, 1000)) 
    plt.ylim(0., 5000.)
    plt.show() 
        

def train(dataloader, model, loss_fn, optimizer, device, l1:float=0.0, l2:float=0.0, pqi:float=0.0):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0.0
    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        if l1 != 0.0: 
            L1_regularizer(model, device, optimizer, l1)
        if l2 != 0.0: 
            L2_regularizer(model, device, optimizer, l2)
        if pqi != 0.0: 
            PQI_regularizer(model, device, optimizer, pqi, 1, 2)
    

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return {"loss" : train_loss / num_batches, 
            "accuracy" : 0.0}
         
            
def test(dataloader, model, loss_fn, device, l1:float=0.0, l2:float=0.0, pqi:float=0.0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() 
            
            if l1 != 0.0: 
                test_loss += l1 * p_norm(model, device, 1)
            if l2 != 0.0: 
                test_loss += l2 * p_norm(model, device, 2) 
            if pqi != 0.0: 
                test_loss += pqi * PQI(model, device, 1, 2)
                
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return {"loss" : test_loss, 
            "accuracy" : correct}