import torch
from torch.utils.data import Dataset

def get_batch(data, split, batch_size, block_size, device):
    data_split = train_data if split == "train" else val_data
    idx = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i : i + block_size] for i in idx])
    y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in idx])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data if split == "train" else val_data, split, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
