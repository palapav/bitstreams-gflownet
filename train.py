import torch
import tqdm
import reward
from gflownet import GFlowNet
from torch.distributions.categorical import Categorical

# via trajectory balance (fixing P_B also)
model = GFlowNet(512)
opt = torch.optim.Adam(model.parameters(), 3e-3)

tb_losses = []
sampled_bit_strings = []
bit_string_freq = {reward.UNBALANCED_REWARD: 0, reward.BALANCED_REWARD: 0, reward.NONE_REWARD: 0}

minibatch_loss = 0
update_freq = 32
print_loss_freq = 1000
logZs = []

for episode in tqdm.tqdm(range(100000), ncols=40):
    reward = None
    is_terminal = False
    state = torch.FloatTensor([0.] * reward.TARGET_BIT_STRING_LEN)

    # total_P_F includes termination probability
    total_P_F = 0
    total_P_B = 0

    while not is_terminal:
        P_F_s = model(state)
        cat = Categorical(logits=P_F_s)
        action = cat.sample()
        total_P_F += cat.log_prob(action)

        # terminal state
        if action == reward.TARGET_BIT_STRING_LEN:
            is_terminal = True
            reward = reward.bits_reward(state)
        else:
            new_state = state.clone()
            new_state[action] = 1.0

            # fixing P_B to arrive at one P_F global minimum via uniform distribution
            fixed_P_B_s = -1 * torch.log(torch.sum(new_state == 1))
            total_P_B += fixed_P_B_s
            state = new_state

    loss = (model.logZ + total_P_F - torch.log(torch.tensor(reward)) - total_P_B).pow(2)
    minibatch_loss += loss

    if episode % update_freq == 0:
        tb_losses.append(minibatch_loss.item())
        minibatch_loss /= update_freq
        minibatch_loss.backward()
        opt.step()
        opt.zero_grad()
        minibatch_loss = 0
        logZs.append(model.logZ.item())

    if episode % print_loss_freq == 0:
          print(f"\nminibatch loss {episode}:\n{tb_losses[-1]}")
          # take an average later
          if tb_losses[-1] < 1e-3: break