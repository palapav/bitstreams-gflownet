import torch
import torch.nn as nn

def bits_to_tensor(bits_arr):
    tensor = torch.tensor(bits_arr, dtype=torch.uint8)


