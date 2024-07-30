import torch
import torch.nn as nn

class ControllerNetwork(nn.Module):
    def __init__(self, num_state, num_reference, num_control, Np, acc_min, acc_max, delta_f_min, delta_f_max):
        super(ControllerNetwork, self).__init__()
        self.num_state = num_state
        self.num_reference = num_reference
        self.num_control = num_control
        self.Np = Np
        self.acc_min = acc_min
        self.acc_max = acc_max
        self.delta_f_min = delta_f_min
        self.delta_f_max = delta_f_max

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.num_state + self.num_reference, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=self.num_control * self.Np),
        )

    def forward(self, features):
        initial_state = features[:, :self.num_state]
        reference = features[:, self.num_state:self.num_state + self.num_reference]

        outputs = self.fc(features)

        outputs = torch.tanh(outputs)

        acc = outputs[..., :self.Np] * (self.acc_max - self.acc_min) / 2 + (self.acc_max + self.acc_min) / 2
        delta_f = outputs[..., self.Np:] * (self.delta_f_max - self.delta_f_min) / 2 + (self.delta_f_max + self.delta_f_min) / 2

        control_sequence = torch.cat((acc, delta_f), dim=1)
        control_sequence = control_sequence.view(-1, self.num_control, self.Np)
        control_sequence = control_sequence.transpose(1, 2)

        return initial_state, control_sequence, reference
