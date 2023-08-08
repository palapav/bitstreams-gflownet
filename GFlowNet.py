import torch
import torch.nn as nn

# train using a flow-matching loss
# let's hardcode for size 12 bit strings -> ideally we would want variable
# size -> 1 for the presence of 1, 0 for the lack of presence of 1, 2 if position hasn't been encountered yet


class GFlowNet(nn.Module):
    def __init__(self, num_hid):
        # edit architecture structure later
        # is nn.Linear only binary values?
        # 12 length of our input vector
        # 2 is the number of child actions stemming from 
        # the current bit string (either place 0 or 1 in next place)
        self.mlp = nn.Sequential(nn.Linear(12, num_hid), nn.LeakyReLU(),
                                 nn.Linear(num_hid, 2))
    
    # x represents the parent bit string
    # will the ouput be probabilistic or definite?
    def forward(self, x):
        # we need to give 0 flow to actions we can't take -> not necessary b/c
        # at a given state we can
        # no need to multiply by (1 - x) -> no illegal actions 
        F = self.mlp(x).exp()
        return F




