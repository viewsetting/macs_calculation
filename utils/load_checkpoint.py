import torch

def load_checkpoint(path,key=None):
    if key is None:
        return torch.load(path)
    else:
        return torch.load(path)[key]