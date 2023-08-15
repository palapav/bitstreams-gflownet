from GFlowNet import GFlowNet
from nn_utils import bits_reward, bits_to_tensor, parent_state_action, TARGET_BIT_STRING_LEN
import torch
from torch.distributions.bernoulli import Bernoulli
import tqdm
import sys

# can build a trainer class

def train():
    # F[s][a] -> outputs the flow to every child state from the parent state
    # indexed at s and action a
    F_sa = GFlowNet(512)
    opt = torch.optim.Adam(F_sa.parameters(), 3e-4)

    losses = []
    sampled_bit_strings = []

    minibatch_loss = 0
    update_freq = 4

    num_episodes = 50,000
    # check on tqdm

    # for checking progress per episode
    # why isn't variable working in range function?
    for episode in range(100):
        # each episode state starts with an empty state
        state = []
        state = bits_to_tensor(state)

        print(f"statekk:\n{state}")

        # Predict F[s][a]
        # tensor of 2 values
        edge_flow_prediction = F_sa(state)

        # 12 flows from parent to children to create 12 bit string
        for t in range(12):
            print(f"t iteration: {t}")
            # normalizing the policy
            policy = edge_flow_prediction / edge_flow_prediction.sum()
            print(f"policy: {policy[1]}")
            # sample action (either 0 or 1) with policy probs
            # the first element in policy vector for 0, second one for 1
            action = Bernoulli(probs=policy[1]).sample()
            print(f"action:{action}")
            print(f"type of action: {type(action)}")

            # Go to the next state:
            print(f"statez: {state}")
            print(f"actionz: {torch.FloatTensor(action)}")
            
            # need to figure out better indexing method
            action_index = torch.where(state == 2)[0][0].item()
            print(f"action index {action_index}")
            print(f"first index: {action_index}")
            new_state = state.clone()
            new_state[action_index] = action

            # enumerate over the parents
            parent_state, parent_action = parent_state_action(new_state)
            # getting the right edge flow based on parent_state/parent_action (based on which binary action the parent took)
            # 1 x 1 tensor
            print(f"parent_state: {parent_state}")
            print(f"parent_action: {parent_action}")

            # in edge flow -> [action 0 flow, action 1 flow]
            parent_action = int(parent_action)
            parent_edge_flow_pred = F_sa(parent_state)[parent_action]
            
            if t == TARGET_BIT_STRING_LEN:
                # edge flow prediction = reward, terminal state
                reward = bits_reward(new_state)
                edge_flow_prediction = reward
            else:
                # reward = 0
                # 1 x 2 arr
                edge_flow_prediction = F_sa(new_state)
            
            # basic loss equation
            # parent edge flow = terminal reward + child edge flow
            # reward is the edge flow from the terminal state to the sink state
            # more clean -> if/else
            # look into theory behind flow-mismatch more
            flow_mismatch = (parent_edge_flow_pred - edge_flow_prediction.sum()) ** 2
            minibatch_loss += flow_mismatch
            # continue iterating to next flow states
            state = new_state
        
        # adding terminal bit string objects to sampled_bit_strings
        sampled_bit_strings.append(state)
        # episode completed, add bit string to list,
        # take the gradient step if finished batch of episodes for training
        if episode % update_freq == 0:
            print(f"minibatch loss: {minibatch_loss}")
            losses.append(minibatch_loss)
            # minibatch loss needs to be in the form of a tensor
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
    
    return losses, sampled_bit_strings


# training a gflownet to match a reward function
# only learning edge flow right now
# explicit policy for taking actions
# start with detail balance, then trajectory balance
# rewrite this with explicit policy
def validate_train(sampled_bit_strings, bits_reward): pass
    # count the number of strings that are palindrome and balanced
    # just palindrome

# plotting loss and generated bit strings
def main():
    losses, sampled_bit_strings = train()
    print(f"losses:\n{losses}")
    # validate_train(sampled_bit_strings, bits_reward)

# unit testing
if __name__ == "__main__":
    main()