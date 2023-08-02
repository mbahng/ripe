from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os 


# the normalization parameters for each dataset 
data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}

def fetch_dataset(name:str, verbose=True): 
    
    dataset = {} 
    
    if verbose:
        print(f'fetching data {name}...')
            
    root = os.path.join('src/datasets')

    if name == "MNIST": 
        
        dataset["train"] = datasets.MNIST(
            root=root,            
            train=True,            
            download=True,          
            transform=transforms.Compose([transforms.ToTensor()])    
        )
        
        dataset["test"] = datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])  
        )
        
        dataset['train'].transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[name])])
        dataset['test'].transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[name])])
        
    elif name == "FashionMNIST": 
        dataset["train"] = datasets.FashionMNIST(
            root=root,            
            train=True,            
            download=True,          
            transform=transforms.Compose([transforms.ToTensor()])    
        )
        
        dataset["test"] = datasets.FashionMNIST(
            root=root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])  
        )
        
        dataset['train'].transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[name])])
        dataset['test'].transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[name])])

    elif name == "CIFAR10": 

        dataset["train"] = datasets.CIFAR10(
            root=root,            
            train=True,            
            download=True,          
            transform=transforms.Compose([transforms.ToTensor()])    
        )
        dataset["test"] = datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])   
        )
        
        dataset['train'].transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[name])])
        
        dataset['test'].transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[name])])
    
    elif name == "CIFAR100": 
        dataset["train"] = datasets.CIFAR100(
            root=root,            
            train=True,            
            download=True,          
            transform=transforms.Compose([transforms.ToTensor()])    
        )
        dataset["test"] = datasets.CIFAR100(
            root=root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])   
        )
        
        dataset['train'].transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[name])])
        
        dataset['test'].transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[name])])
    
    elif name == "SVHN": 
        dataset["train"] = datasets.SVHN(
            root=root,            
            train=True,            
            download=True,          
            transform=transforms.Compose([transforms.ToTensor()])    
        )
        dataset["test"] = datasets.SVHN(
            root=root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])   
        )
        
        dataset['train'].transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[name])])
        
        dataset['test'].transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[name])])
    
    else: 
        raise ValueError("That is not a valid name. ")
    
    if verbose:
        print('data ready')
    
    return dataset["train"], dataset["test"]
        

def dataloader(training_data, test_data, batch_size=-1): 
    
    if batch_size == -1: 
    
        train_dataloader = DataLoader(training_data,  
                                        batch_size=len(training_data),    
                                        shuffle=True     
                                        )
        test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)
    
    else: 
        train_dataloader = DataLoader(training_data,  
                                        batch_size=batch_size,    
                                        shuffle=True     
                                        )
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader