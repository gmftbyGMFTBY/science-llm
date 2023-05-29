from header import *

# monitor
def monitor_token_acc(logits, labels):
    # calculate the token accuarcy for monitor
    chosen_tokens = torch.max(logits, dim=-1)[1]
    gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
    valid_mask = (labels != -100).reshape(-1)
    valid_tokens = gen_acc & valid_mask
    gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
    return gen_acc
