import torch

TARGET_BIT_STRING_LEN = 12

def is_balanced(bits_tensor):
    num_one_bits = torch.sum(bits_tensor == 1)
    num_zero_bits = torch.sum(bits_tensor == 0)
    return torch.equals(num_one_bits, num_zero_bits)

def is_palindrome(bits_tensor):
    # dim=0 along rows
    reversed_bits = torch.flip(bits_tensor, dims=[0])
    return torch.equals(bits_tensor, reversed_bits)

# input must be tensor
def bits_reward(bits_tensor):
    if not isinstance(bits_tensor, torch.FloatTensor): raise TypeError("GFlowNet state is not of type tensor")

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
def parent_state_action(state):
    if not isinstance(state, torch.FloatTensor): raise TypeError("GFlowNet state is not of type tensor")
    # numpy and pytorch have similar operations
    # we figure out where the first 2 starts
    # parent action -> How do we get from that parent state to child state
    # parent_state, parent_action = [], None

    # print(f"statekkk: {state}")

    # need to fix the double zero, find better function
    # first empty index is at least at 1 -> because we are always guarenteed
    # at a child state
    # last child state in DAG (initial implementation -> tree)
    # parent action -> leads to child state
    parent_action_index = 11
    if torch.where(state == 2)[0].numel() > 0:
        first_empty_index = torch.where(state == 2)[0][0].item()
        parent_action_index = first_empty_index - 1

    parent_action = state[parent_action_index].item()
    # is the clone necessary?
    parent_state = state.clone()
    # harcode with misc number (not 0, 1)
    parent_state[parent_action_index] = 2
    
    return parent_state, parent_action

# prepares for neural network input
def bits_to_tensor(bits_state):
    if not isinstance(bits_state, list): raise TypeError("GFlowNet state is not of type list")
    bits_list_length = len(bits_state)
    if bits_list_length > TARGET_BIT_STRING_LEN: raise ValueError(f"bit string length is greater than {TARGET_BIT_STRING_LEN}")

    # need to figure out a way to stop -> torch.full
    empty_slots = [2 for i in range(0, TARGET_BIT_STRING_LEN - bits_list_length)]
    # print(f"bits before state: {bits_state}")
    bits_state.extend(empty_slots)
    # print(f"bitzz state: {bits_state}")
    input_tensor = torch.FloatTensor(bits_state)
    # print(f"input tensor:{input_tensor}")
    return input_tensor

def main():
    # use a test framework
    # test cases for is_balanced()

    # test cases for is_palindrome()

    # test cases for bits_reward()

    # test cases for parent_state()

    # test cases for bits_to_tensor()
    bits_state = [0, 0, 1]
    input_tensor1 = bits_to_tensor(bits_state)

# unit testing
if __name__ == "__main__":
    main()

