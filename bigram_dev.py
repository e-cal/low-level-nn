"""Andrej Karpathy's bigram model from his GPT tutorial."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
#                                   Raw data loading
# ==============================================================================

# Download data
if not os.path.exists("data/shakespeare.txt"):
    print("Downloading Shakespeare...")
    os.system(
        "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/shakespeare.txt"
    )

# Load data
with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("loaded text")
print("corpus length:", len(text))

# Metadata
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("chars: \\n ␣", " ".join(chars[2:]))
print("vocab size:", vocab_size)

# Encoding and decoding
stoi = {ch: i for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
itos = {i: ch for i, ch in enumerate(chars)}
decode = lambda x: "".join([itos[i] for i in x])

print("encoding and decoding sanity check:")
enc = encode(" hello ")
dec = decode(enc)
print(f"'{dec}' -> {enc}")
print()

# To tensor and proper loading
print("loading into a tensor")
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print()

n = int(0.9 * len(data))  # 90% train, 10% val, no shuffle
train_data = data[:n]
val_data = data[n:]

block_size = 8

print("illustrating how training sequences (context is 1-block_size)")
x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"input: {context}, target: {target}")
print()

# ==============================================================================
#                                   Data Loader
# ==============================================================================

torch.manual_seed(1337)
batch_size = 4
block_size = 8


def get_batch(data):
    # get `batch_size` random indecies
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


print("testing get_batch")
xb, yb = get_batch(train_data)
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

for b in range(batch_size):  # batch
    print(f"sample {b+1}")
    for t in range(block_size):  # time
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"input: {context}, target: {target}")


# ==============================================================================
#                                 Bigram Model
# ==============================================================================
print()
print("Bigram model")


class Bigram(nn.Module):
    def __init__(self, vocab_size):
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


model = Bigram(vocab_size)
logits, loss = model(xb, yb)
print("logits shape:", logits.shape)
print("loss:", loss)

print(
    decode(
        model.generate(idx=torch.zeros(1, 1, dtype=torch.long), max_tokens=100)[0].tolist()  # fmt: skip
    )
)

print()
print("training")
batch_size = 32
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
for i in range(10000):
    optimizer.zero_grad()
    x, y = get_batch(train_data)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    # if i % 100 == 0:
    #     print(f"step {i}, loss {loss.item():.2f}")

print(f"step {i}, loss {loss.item():.2f}")  # type: ignore


print(
    decode(
        model.generate(idx=torch.zeros(1, 1, dtype=torch.long), max_tokens=100)[0].tolist()  # fmt: skip
    )
)
