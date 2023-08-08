from GFlowNet import GFlowNet
import torch
from torch.distributions.categorical import Categorical
import tqdm

# F[s][a] -> outputs the flow to every child state from the parent state
# indexed at s and action a
F_sa = GFlowNet(512)
opt = torch.optim.Adam(F_sa.parameters(), 3e-4)

minibatch_loss = 0
update_freq = 4

# for checking progress per episode
for episode in tqdm.tqdm(range(50000), ncols=40):
    # each episode state starts with an empty state
    state = []

    # Predict F[s][a]
    edge_flow_prediction = 

    pass


