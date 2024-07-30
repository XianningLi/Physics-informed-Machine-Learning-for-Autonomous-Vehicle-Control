import torch
import torch.nn as nn
from DPCModel.ControllerNetwork import ControllerNetwork
from DPCModel.VehicleDynamics import vehicle_dynamics
from DPCModel.VehicleDynamicsNetwork import VehicleDynamicsNetwork
from DPCModel.CustomLoss import CustomLoss
from DPCModel.IntegratedNetwork import IntegratedNetwork
from DPCModel.SetSeed import set_seed
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

set_seed()

params = {
    'Cf': torch.tensor(1250),
    'Cr': torch.tensor(755),
    'a': torch.tensor(1.015),
    'b': torch.tensor(1.895),
    'm': torch.tensor(1270),
    'Iz': torch.tensor(1536.7),
}
dt = torch.tensor(0.5)
Np = 10

controller_params = {
    'num_state': 6,
    'num_reference': 2,
    'num_control': 2,
    'Np': Np,
    'acc_min': -3.0,
    'acc_max': 3.0,
    'delta_f_min': -0.3,
    'delta_f_max': 0.3
}

integrated_net = IntegratedNetwork(controller_params, params, dt, Np)
print(integrated_net)

train_data = pd.read_csv('Train Data.csv').values
test_data = pd.read_csv('Test Data.csv').values
print(test_data)

train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)

train_dataset = TensorDataset(train_tensor)
test_dataset = TensorDataset(test_tensor)

train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=150, shuffle=False)

model = IntegratedNetwork(controller_params, params, dt, Np).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

Qx_values = [0, 60, 500, 50, 10, 100]
Qu_values = [5, 500]
Qt_values = [0, 60, 1000, 70, 20, 200]

loss_function = CustomLoss(model, Qx_values, Qu_values, Qt_values, device)

train_losses = []
test_losses = []

def train(epoch):
    model.train()
    total_loss = 0
    time_start = time.time()
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        states, initial_state, reference, control_sequence = model(data)
        loss = loss_function(states, initial_state, reference, control_sequence)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch {epoch}: Training loss: {avg_loss}')
    time_end = time.time()
    print(f'Training time: {time_end - time_start}')

def test():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(test_loader):
            data = data.to(device)
            states, initial_state, reference, control_sequence = model(data)
            loss = loss_function(states, initial_state, reference, control_sequence)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    test_losses.append(avg_loss)
    print(f'Test loss: {avg_loss}')

for epoch in range(1, 1000):
    train(epoch)
    test()

torch.save(model, 'complete_integrated_net.pth')
torch.save(model.controller, 'controller_net.pth')

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Test Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

data_to_save = {
    'train_losses': train_losses,
    'test_losses': test_losses
}

with open('losses.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)
