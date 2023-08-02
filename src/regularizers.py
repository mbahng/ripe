import torch 
import math


def p_norm(model, device, p:int, normalize=True): 
    p_norm = None
    N = 0
    for W in model.parameters(): 
        if p_norm == None: 
            p_norm = W.norm(p) ** p
        else: 
            p_norm += W.norm(p) ** p
            
        if normalize == True: 
            N += math.prod(W.size())
            
    if normalize == True: 
        return (1/N) * p_norm.to(device)
    else: 
        return p_norm.to(device)
    
def PQI(model, device, p, q): 
    
    d = sum(math.prod(param.size()) for param in model.parameters())
    pq = d ** ((1/q) - (1/p)) * (p_norm(model, device, p)/p_norm(model, device, q))
    
    return pq

def L1_regularizer(model, device, optimizer, lmbda): 
    # L1 regularizer with clipping 
    
    positives1 = []
    negatives1 = []
    for W in model.parameters(): 
        mask1 = torch.where(W.data > 0, 1.0, 0.0)
        mask2 = torch.where(W.data < 0, 1.0, 0.0)
        positives1.append(mask1) 
        negatives1.append(mask2)
        
    reg_loss = lmbda * p_norm(model, device, 1, normalize=True)
    reg_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    positives2 = [] 
    negatives2 = [] 
    for W in model.parameters(): 
        mask1 = torch.where(W.data > 0, 1.0, 0.0)
        mask2 = torch.where(W.data < 0, 1.0, 0.0)
        positives2.append(mask1) 
        negatives2.append(mask2)
        
    clip_mask = [] 
    for i in range(len(positives1)): 
        clip = positives1[i] * positives2[i] + negatives1[i] * negatives2[i]
        clip_mask.append(clip)
    
    with torch.no_grad(): 
        i = 0
        for name, param in model.named_parameters(): 
            param.copy_(clip_mask[i] * param)
            i += 1 
            

def L2_regularizer(model, device, optimizer, lmbda): 
    # L2 regularizer without clipping 
    
    reg_loss = lmbda * p_norm(model, device, 2, normalize=True)
    reg_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
def PQI_regularizer(model, device, optimizer, lmbda, p, q): 
    reg_loss = lmbda * PQI(model, device, p, q)
    reg_loss.backward()
    optimizer.step()
    optimizer.zero_grad()