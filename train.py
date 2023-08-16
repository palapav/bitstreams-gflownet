from GFlowNet import GFlowNet
from nn_utils import bits_reward, bits_to_tensor, parent_state_action, TARGET_BIT_STRING_LEN
import torch
from torch.distributions.bernoulli import Bernoulli
import tqdm
import sys

def train():
    # F[s][a] -> outputs the flow to every child state from the parent state
    F_sa = GFlowNet(512)
    opt = torch.optim.Adam(F_sa.parameters(), 3e-4)

    losses = []
    sampled_bit_strings = []

    minibatch_loss = 0

    # make batch size higher
    update_freq = 4

    num_episodes = 50,000

    for episode in range(3000):
        # each episode state starts with an empty state
        state = []
        state = bits_to_tensor(state)

        # tensor[ , ] = F[s][a]
        edge_flow_prediction = F_sa(state)

        # 12 flows from parent to children to create 12 bit string
        for t in range(TARGET_BIT_STRING_LEN):
            policy = edge_flow_prediction / edge_flow_prediction.sum()

            # the first element in policy vector for 0, second one for 1
            action = Bernoulli(probs=policy[1]).sample()

            # need to figure out better indexing method
            action_index = torch.where(state == 2)[0][0].item()
            new_state = state.clone()
            new_state[action_index] = action

            # enumerate over the parents
            parent_state, parent_action = state.clone(), action

            # in edge flow -> [action 0 flow, action 1 flow]
            parent_action = int(parent_action)
            parent_edge_flow_pred = F_sa(parent_state)[parent_action]

            if t == TARGET_BIT_STRING_LEN:
                # edge flow prediction = reward, terminal state
                reward = bits_reward(new_state)
                # print(f"my reward: {reward}")
                edge_flow_prediction = reward
            else:
                # reward = 0
                # 1 x 2 arr
                edge_flow_prediction = F_sa(new_state)

            flow_mismatch = (parent_edge_flow_pred - edge_flow_prediction.sum()) ** 2
            minibatch_loss += flow_mismatch

            # continue iterating to next flow states
            state = new_state
        
        # adding terminal bit string objects to sampled_bit_strings
        sampled_bit_strings.append(state)

        if episode % update_freq == 0:
            print(f"minibatch loss {episode}:\n{minibatch_loss}")
            losses.append(minibatch_loss)
            # minibatch loss needs to be in the form of a tensor
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
    
    return losses, sampled_bit_strings


def validate_train(sampled_bit_tensors, bits_reward):
    bit_strings_type = {"palindrome+balanced": 0, "palindrome": 0, "balanced": 0, "none": 0}

    for bits_tensor in sampled_bit_tensors:
        # print(f"bits tensor in eval: {bits_tensor}")
        reward = bits_reward(bits_tensor)
        # print(f"reward from sampled bits: {reward}")
        if reward == 4:
            bit_strings_type["palindrome+balanced"] = bit_strings_type.get("palindrome+balanced") + 1
        elif reward == 2:
            bit_strings_type["palindrome"] = bit_strings_type.get("palindrome") + 1
        elif reward == 1:
            bit_strings_type["balanced"] = bit_strings_type.get("balanced") + 1
        else:
            bit_strings_type["none"] = bit_strings_type.get("none") + 1

    print(f"Number of palindrome strings: {bit_strings_type.get('palindrome')}")
    print(f"Number of balanced strings: {bit_strings_type.get('balanced')}")
    print(f"Number of palindrome + balanced strings: {bit_strings_type.get('palindrome+balanced')}")
    print(f"Number of None strings: {bit_strings_type.get('none')}")
    print(f"balanced to palindrome ratio: {bit_strings_type.get('palindrome+balanced') / bit_strings_type.get('balanced')}")

# plotting loss and generated bit strings
def main():
    losses, sampled_bit_strings = train()
    validate_train(sampled_bit_strings, bits_reward)

# unit testing
if __name__ == "__main__":
    main()