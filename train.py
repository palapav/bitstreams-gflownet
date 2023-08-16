from GFlowNet import GFlowNet
from nn_utils import bits_reward, bits_to_tensor, TARGET_BIT_STRING_LEN
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

    for episode in range(50000):
        # each episode state starts with an empty state
        state = []
        # state = bits_to_tensor(state)
        state = torch.FloatTensor([2.] * TARGET_BIT_STRING_LEN)

        # tensor[ , ] = F[s][a]
        edge_flow_prediction = F_sa(state)
        # print(f"beginning: {edge_flow_prediction}")

        # 6 flows from parent to children to create 6 bit string
        for t in range(TARGET_BIT_STRING_LEN):
            policy = edge_flow_prediction / edge_flow_prediction.sum()

            # print(f"Policy in {t}: {policy[1]}")

            # the first element in policy vector for 0, second one for 1
            action = Bernoulli(probs=policy[1]).sample()
            # print(f"action: {action}")

            # need to figure out better indexing method
            action_index = torch.where(state == 2)[0][0].item()
            new_state = state.clone()
            new_state[action_index] = action

            # enumerate over the parents
            parent_state, parent_action = state.clone(), action

            # in edge flow -> [action 0 flow, action 1 flow]
            parent_action = int(parent_action)
            parent_edge_flow_pred = F_sa(parent_state)[parent_action]

            reward = 0
            edge_flow_prediction = torch.zeros(2)
            if t == TARGET_BIT_STRING_LEN - 1:
                # print("I AM GETTING A REWARD:")
                reward = bits_reward(new_state)
            else: edge_flow_prediction = F_sa(new_state)

            flow_mismatch = (parent_edge_flow_pred - edge_flow_prediction.sum() - reward) ** 2
            minibatch_loss += flow_mismatch

            state = new_state
        
        # adding terminal bit string objects to sampled_bit_strings
        sampled_bit_strings.append(state)
        if episode % 10000 == 0:
            validate_train(sampled_bit_strings, bits_reward)

        if episode % update_freq == 0:
            print(f"minibatch loss {episode}:\n{minibatch_loss.item()}")
            losses.append(minibatch_loss.item())
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
        # if reward == 4:
        #     bit_strings_type["palindrome+balanced"] = bit_strings_type.get("palindrome+balanced") + 1
        if reward == 2:
            bit_strings_type["palindrome"] = bit_strings_type.get("palindrome") + 1
        elif reward == 1:
            bit_strings_type["balanced"] = bit_strings_type.get("balanced") + 1
        else:
            bit_strings_type["none"] = bit_strings_type.get("none") + 1

    print(f"Number of palindrome strings: {bit_strings_type.get('palindrome')}")
    print(f"Number of balanced strings: {bit_strings_type.get('balanced')}")
    # print(f"Number of palindrome + balanced strings: {bit_strings_type.get('palindrome+balanced')}")
    print(f"Number of None strings: {bit_strings_type.get('none')}")
    # print(f"balanced to palindrome ratio: {bit_strings_type.get('palindrome+balanced') / bit_strings_type.get('balanced')}")

# plotting loss and generated bit strings
def main():
    losses, sampled_bit_strings = train()
    validate_train(sampled_bit_strings, bits_reward)

# unit testing
if __name__ == "__main__":
    main()