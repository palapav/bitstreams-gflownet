import torch
import torch.nn as nn
from reward import TARGET_BIT_STRING_LEN

class GFlowNet(nn.Module):
    def __init__(self, num_hid):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(TARGET_BIT_STRING_LEN, num_hid), nn.LeakyReLU(),
                                 nn.Linear(num_hid, TARGET_BIT_STRING_LEN + 1))

        self.logZ = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.size(0) != TARGET_BIT_STRING_LEN: raise AssertionError("Input tensor of incorrect length")

        P_F = self.mlp(x)

        # masking illegal actions
        P_F[torch.where(x == 1.0)] = -100
        return P_F