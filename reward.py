import torch

# constants
TARGET_BIT_STRING_LEN = 8
PALINDROME_REWARD = 1
BALANCED_REWARD = 10
NONE_REWARD = 0
UNBALANCED_REWARD = 1

def is_balanced(bits_tensor):
    """returns a boolean value for whether the given bits_tensor is
    has an equal number of 0s and 1s or not"""
    num_one_bits = torch.sum(bits_tensor == 1)
    num_zero_bits = torch.sum(bits_tensor == 0)
    return torch.equal(num_one_bits, num_zero_bits)

def is_palindrome(bits_tensor):
    # dim=0 along rows
    reversed_bits = torch.flip(bits_tensor, dims=[0])
    return torch.equal(bits_tensor, reversed_bits)

# input must be tensor
def bits_reward(bits_tensor):
    if not isinstance(bits_tensor, torch.FloatTensor): raise TypeError("GFlowNet state is not of type tensor")

    if is_balanced(bits_tensor): return BALANCED_REWARD

    return UNBALANCED_REWARD

bits_reward(torch.tensor([0.0, 0.0]))
bits_reward(torch.tensor([0.0, 1.0]))