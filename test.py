import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
# first_tensor = torch.tensor([0, 1, 2, 0, 1, 2, 2, 2])

# twos_tensor = torch.FloatTensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])

# ones_tensor = torch.FloatTensor([1., 1., 1., 1., 1.])
# print(torch.where(ones_tensor == 2)[0].numel())

# print(torch.where(twos_tensor == 2.0)[0][0].item())

# test_first = first_tensor[0].item()
# # print(test_first)

# layer1 = nn.Linear(4, 50)

# # sec_tensor = torch.LongTensor([1, 2, 3, 4])
# # nn.Linear expects float tensor
# sec_tensor = torch.randn(1, 4)

# output = layer1(sec_tensor)

# # print(sec_tensor.dtype)

# # print(not torch.any((first_tensor != 0) & (first_tensor != 1)).item())

# # print(torch.equal(torch.sum(first_tensor), torch.sum(sec_tensor)))

# # print(type(torch.equal(first_tensor, sec_tensor)))


# test = torch.tensor([2., 1.])

# print(torch.any(test == 2).item())

# test[0] = 0.000001

# binary_output = Bernoulli(probs=test[0]).sample()

# state = torch.FloatTensor([])

# print(binary_output)

terminal_state = torch.FloatTensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.])

init_state = torch.FloatTensor([2.] * 6)
print(init_state)
# print(torch.any(terminal_state != 2.0).item())

