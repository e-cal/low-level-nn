"""Based on Andrej Karpathy's bigram model from his GPT tutorial."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
#                                   Data Loader
# ==============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
block_size = 8
global vocab_size

torch.manual_seed(1337)


def get_text():
    # Download data
    if not os.path.exists("data/shakespeare.txt"):
        print("Downloading Shakespeare...")
        os.system(
            "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/shakespeare.txt"
        )

    # Read text
    with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    return text


def get_batch(data):
    # get `batch_size` random indecies
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # (B,T,C) -> (batch_size=4, block_size=8, vocab_size=65)
        logits: torch.Tensor = self.token_embedding(idx)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # channels need to be second dimension
            targets = targets.view(-1)  # flatten to match logits
            loss = F.cross_entropy(logits, targets)  # nll loss

        return logits, loss

    def generate(self, idx, max_tokens):
        # idx is a (B,T) tensor of indices in the current context

        for _ in range(max_tokens):
            logits, loss = self(idx)
            # focus on only the last time step
            logits = logits[:, -1, :]  # -> (B,C)
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat([idx, idx_next], dim=1)  # (B,T+1)

        return idx

    def train(self, train_data, val_data, epochs=10, lr=0.001):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        eval_interval = epochs // 10
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            x, y = get_batch(train_data)
            _, loss = self(x, y)
            loss.backward()
            optimizer.step()

            if epoch % eval_interval == 0:
                losses = torch.zeros(5)
                for i in range(5):
                    x, y = get_batch(val_data)
                    _, losses[i] = self(x, y)
                print(
                    f"epoch {epoch}, loss {loss.item():.2f}, val loss {losses.mean().item():.2f}"
                )


if __name__ == "__main__":
    # get text
    text = get_text()
    chars = sorted(list(set(text)))
    global vocab_size
    vocab_size = len(chars)

    # Character encoding and decoding
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda x: [stoi[ch] for ch in x]
    itos = {i: ch for i, ch in enumerate(chars)}
    decode = lambda x: "".join([itos[i] for i in x])

    # load data
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # 90% train, 10% val, no shuffle
    train_data, val_data = data[:n], data[n:]

    model = Bigram().to(device)
    model.train(train_data, val_data, epochs=10000)

    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(ctx, max_tokens=500)[0].tolist()))
