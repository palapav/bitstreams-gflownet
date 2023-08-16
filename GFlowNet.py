import torch
import torch.nn as nn
from nn_utils import TARGET_BIT_STRING_LEN

# train using a flow-matching loss
# let's hardcode for size 12 bit strings -> ideally we would want variable
# size -> 1 for the presence of 1, 0 for the lack of presence of 1, 2 if position hasn't been encountered yet
class GFlowNet(nn.Module):
    def __init__(self, num_hid):
        # super should be first line
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(TARGET_BIT_STRING_LEN, num_hid), nn.LeakyReLU(),
                                 nn.Linear(num_hid, 2))
    
    # x represents the parent bit string
    # will the ouput be probabilistic or definite?
    def forward(self, x):
        # we need to give 0 flow to actions we can't take -> not necessary b/c
        # no need to multiply by (1 - x) -> no illegal actions
        # instead of .exp -> try log
        F = self.mlp(x).exp()
        return F




