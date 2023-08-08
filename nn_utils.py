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


# plan out function -> robustness
def parent_state(state):
    if not isinstance(state, torch.Tensor): raise TypeError("GFlowNet state is not of type tensor")
    # numpy and pytorch have similar operations
    # we figure out where the first 2 starts
    # parent action -> How do we get from that parent state to child state
    parent_state, parent_action = None

    first_empty_index = torch.argmax(state == 2)

    # no parent states or actions exist yet
    # can we reduce the number of return statements?
    if first_empty_index == 0: return parent_state, parent_action

    parent_action = state[first_empty_index - 1].item()
    if first_empty_index == 1:
        return parent_state, parent_action
    
    parent_state = state[0:first_empty_index - 2]
    return parent_state, parent_action
    
    # find the first occurence of where the 2 occurs
    # subtract 1 in index -> child action at index + state in totality, 
    # subtract 1 in index -> parent state

    # edge case when 2 is the first index

def main(): pass

# unit testing
if __name__ == "__main__":
    main()

