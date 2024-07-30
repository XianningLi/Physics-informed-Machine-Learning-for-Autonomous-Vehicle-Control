from DPCModel.VehicleDynamics import vehicle_dynamics
import torch
import torch.nn as nn

class VehicleDynamicsNetwork(nn.Module):
    def __init__(self, params, dt, prediction_horizon):
        super(VehicleDynamicsNetwork, self).__init__()
        self.params = params
        self.dt = dt
        self.prediction_horizon = prediction_horizon

    def forward(self, initial_state, control_sequence, reference):
        states = [initial_state]
        for i in range(self.prediction_horizon):
            next_state = vehicle_dynamics(states[-1], control_sequence[:, i, :], self.params, self.dt)
            states.append(next_state)
        return torch.stack(states, dim=1), initial_state, reference, control_sequence
