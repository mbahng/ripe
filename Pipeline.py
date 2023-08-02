import torch 
import torch.nn as nn 
import os 
from pprint import pprint
from src.device import select_device 
from src.testtrain import * 
from src.data import fetch_dataset, dataloader
from src.models.mlp import MLP 
from src.models.linear import Linear

torch.manual_seed(0)

device = select_device()

training_data, test_data = fetch_dataset("MNIST", verbose=False)

train_dataloader, test_dataloader = dataloader(
    training_data, test_data, 
    batch_size=-1)


model = Linear(
    data_shape = (28, 28, ), 
    target_size = 10
    ).to(device)

# model = MLP(
#     data_shape=(28, 28), 
#     hidden_size=256, 
#     scale_factor=1, 
#     num_layers=3, 
#     activation="relu", 
#     target_size=10
# ).to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),     
    lr=1e-3,                
    momentum=0.9
)

l1 = 0.
l2 = 0. 
pqi=0.

train_losses = []
test_losses = [] 
test_accuracy = []
PQIs = []

epochs = 100
for t in range(epochs): 
    print(f"Epoch {t+1}\n-------------------------------")
    
    train_dict = train(train_dataloader, model, loss_fn, optimizer, device, l1=l1, l2=l2, pqi=pqi)
    test_dict = test(test_dataloader, model, loss_fn, device, l1=l1, l2=l2, pqi=pqi)
    
    train_losses.append(train_dict["loss"])
    test_losses.append(test_dict["loss"])
    test_accuracy.append(test_dict["accuracy"])
    pqi = PQI(model, device, 1, 2).item()
    PQIs.append(pqi)
    
    pprint(f"L0 Sparsity : {100 * L0_sparsity(model)}%")
    pprint(f"PQ Sparsity : {pqi}")
    
    
    # parameterDistribution(model)
    
plt.plot(PQIs, test_accuracy)
plt.xlabel("Sparsity (PQIs)")
plt.ylabel("Test Accuracy")
plt.title("MNIST")
plt.show() 