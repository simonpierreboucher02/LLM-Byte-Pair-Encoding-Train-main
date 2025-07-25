import torch
from torch.optim import AdamW
from accelerate import Accelerator
import os
from tokenizer_utils import train_tokenizer, encode, decode
from training_utils import get_batch, estimate_loss
from model import GPT

if __name__ == "__main__":
    batch_size = 64
    block_size = 256
    max_iters = 200
    eval_interval = 100
    learning_rate = 3e-4
    eval_iters = 200
    n_embed = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    accelerator = Accelerator()
    device = accelerator.device

    with open("pg2554.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = train_tokenizer([text])
    data = torch.tensor(encode(text, tokenizer), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    vocab_size = tokenizer.get_vocab_size()
    model = GPT(vocab_size, n_embed, block_size, n_layer, n_head, dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model, optimizer, train_data = accelerator.prepare(model, optimizer, train_data)

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device)
            print(f"| step {iter}: train loss {losses['train']:.4f} | validation loss {losses['val']:.4f} |")

        xb, yb = get_batch(train_data, "train", batch_size, block_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        optimizer.step()

    model_dir = "./scratchGPT/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = model_dir + "model.pt"
    torch.save(model.state_dict(), model_path)

    loaded_model = GPT(vocab_size, n_embed, block_size, n_layer, n_head, dropout).to(device)
    loaded_model.load_state_dict(torch.load(model_path))

    context = torch.tensor([[tokenizer.encode("The").ids[0]]], dtype=torch.long, device=device)
    generated_ids = loaded_model.generate(context, max_new_tokens=50)[0].tolist()
    print(decode(generated_ids, tokenizer))
