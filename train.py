from GFlowNet import GFlowNet
from nn_utils import bits_reward, bits_to_tensor, parent_state_action, TARGET_BIT_STRING_LEN
import torch
from torch.distributions.bernoulli import Bernoulli
import tqdm

def train():
    # F[s][a] -> outputs the flow to every child state from the parent state
    # indexed at s and action a
    F_sa = GFlowNet(512)
    opt = torch.optim.Adam(F_sa.parameters(), 3e-4)

    losses = []
    sampled_bit_strings = []

    minibatch_loss = 0
    update_freq = 4

    # for checking progress per episode
    for episode in tqdm.tqdm(range(50000), ncols=40):
        # each episode state starts with an empty state
        state = []

        # Predict F[s][a]
        # tensor of 2 values
        edge_flow_prediction = F_sa(bits_to_tensor(state))

        # 12 flows from parent to children to create 12 bit string
        for t in range(12):
            # normalizing the policy
            policy = edge_flow_prediction / edge_flow_prediction.sum()
            # sample action (either 0 or 1)
            action = Bernoulli(probs=policy)

            # Go to the next state:
            new_state = state + [action]

            # enumerate over the parents
            parent_state, parent_action = parent_state_action(torch.tensor(new_state))
            # getting the right edge flow based on parent_state/parent_action (based on which binary action the parent took)
            # 1 x 1 tensor
            # is parent action here necessary?
            parent_edge_flow_pred = F_sa(parent_state)[parent_action]
            
            if t == TARGET_BIT_STRING_LEN:
                reward = bits_reward(torch.tensor(new_state))
                edge_flow_prediction = 0
            else:
                reward = 0
                edge_flow_prediction = F_sa(bits_to_tensor(new_state))
            
            # basic loss equation
            # parent edge flow = terminal reward + child edge flow
            flow_mismatch = (parent_edge_flow_pred - edge_flow_prediction.sum() - reward) ** 2
            minibatch_loss += flow_mismatch
            # continue iterating to next flow states
            state = new_state
        
        # adding terminal bit string objects to sampled_bit_strings
        sampled_bit_strings.append(state)
        # episode completed, add bit string to list,
        # take the gradient step if finished batch of episodes for training
        if episode % update_freq == 0:
            losses.append(minibatch_loss)
            # minibatch loss needs to be in the form of a tensor
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0

# plotting loss and generated bit strings
def main(): pass

# unit testing
if __name__ == "__main__":
    main()