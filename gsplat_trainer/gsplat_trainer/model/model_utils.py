import torch

def inverse_sigmoid(x):
    return torch.log(x/(1-x))