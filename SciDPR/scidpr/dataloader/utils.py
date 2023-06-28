from header import *

def generate_mask(ids, pad_token_idx=0):
    mask = torch.ones_like(ids)
    mask[ids == pad_token_idx] == 0
    return mask
