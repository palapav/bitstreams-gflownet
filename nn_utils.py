import torch

def is_balanced(bits_tensor):
    num_one_bits = torch.sum(bits_tensor == 1)
    num_zero_bits = torch.sum(bits_tensor == 0)
    return torch.equals(num_one_bits, num_zero_bits)

def is_palindrome(bits_tensor):
    # dim=0 along rows
    reversed_bits = torch.flip(bits_tensor, dims=[0])
    return torch.equals(bits_tensor, reversed_bits)

def bits_reward(bits_tensor):
    # ensures bit string has only zeroes and ones
    if not torch.any((bits_tensor != 0) & (bits_tensor != 1)).item(): return 0     
    # bits_tensor that is a palindrome and balanced should appear 2
    # times more than a bits_tensor that is strictly palindrome and 4 times
    # more than palindrome that is strictly balanced
    if is_palindrome(bits_tensor) and is_balanced(bits_tensor): return 4
    if is_palindrome(bits_tensor): return 2
    if is_balanced(bits_tensor): return 1

    # hopefully we don't see any other bit strings in terminal composite objects
    return 0

