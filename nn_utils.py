import torch

TARGET_BIT_STRING_LEN = 4

def is_balanced(bits_tensor):
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

    if torch.any(bits_tensor == 2.0).item():
        # print(f"uncompleted tensor:\n{bits_tensor}") 
        raise ValueError("terminal state is not complete yet")     
    
    # conflicting dependencies in reward function
    if is_palindrome(bits_tensor) and is_balanced(bits_tensor): return 0
    if is_palindrome(bits_tensor): return 2
    if is_balanced(bits_tensor): return 1

    # hopefully we don't see any other bit strings in terminal composite objects
    return 0

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
    terminal_state = torch.FloatTensor([0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0.])
    print(bits_reward(terminal_state))
    print(f"Balanced: {is_balanced(terminal_state)}")
    print(f"Palindrome: {is_palindrome(terminal_state)}")
    print(f"Balanced + Palindrome: {is_balanced(terminal_state) and is_palindrome(terminal_state)}")

    # test cases for is_palindrome()

    # test cases for bits_reward()

    # test cases for parent_state()
    # child_state = torch.FloatTensor([0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0.])
    child_state = torch.FloatTensor([1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2.])
    # parent_state, parent_action = parent_state_action(child_state)
    # print(f"parent state: {parent_state}")
    # print(f"parent action: {parent_action}")


    # test cases for bits_to_tensor()
    # bits_state = [0, 0, 1]
    # input_tensor1 = bits_to_tensor(bits_state)

# unit testing
if __name__ == "__main__":
    main()

