import os

import torch
import torch.nn as nn
import torch.nn.functional as F

n_iters = 5000
lr = 3e-4

batch_size = 32
block_size = 128
n_emb = 384
n_head = 6
n_blocks = 6
dropout = 0.2

out_tokens = 500


global vocab_size
torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu" and (n_blocks > 3 or block_size > 64):
    print("WARNING: this is gonna take a while...")


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


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # not a model parameter
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        v = self.value(x)  # (B,T,hs)

        # compute attention scores (affinites)
        w = q @ k.transpose(-2, -1) * (C**-0.5)  # (B,T,C)⋅(B,C,T) -> (B,T,T)
        w = w.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # (B,T,T)
        w = F.softmax(w, dim=-1)  # (B,T,T)
        w = self.dropout(w)  # (B,T,T)

        # perform weighted aggregation of the values
        out = w @ v  # (B,T,T)⋅(B,T,C) -> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size)] * num_heads)
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_emb) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),  # projection back to residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attentian) + computation (feedforward)"""

    def __init__(self, n_emb, n_head) -> None:
        super().__init__()
        head_size = n_emb // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        # having `x + ...` makes it a residual (skip) connection
        # during training, the gradient "flows" through the skip connection
        # forking identically, so there is something like a "gradient highway"
        # straing from the input to the output (i.e. skipping the residual pathway)
        # NOTE: this is a huge improvement, need to learn more about it
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_emb)
        self.pos_embedding = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, n_head=n_head)] * n_blocks)
        self.ln = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape  # = targets.shape
        tok_emb = self.token_embedding(idx)  # (B,T,C)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

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
            # clip to the last block_size tokens
            # (careful not to overwrite og idx since it gets concatenated and used as output)
            idx_clipped = idx[:, -block_size:]  # -> (B,T)
            logits, loss = self(idx_clipped)
            # focus on only the last time step
            logits = logits[:, -1, :]  # -> (B,C)
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat([idx, idx_next], dim=1)  # (B,T+1)

        return idx

    def train(self, train_data, val_data, iters=n_iters, lr=0.001):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        eval_interval = iters // 10
        for epoch in range(iters):
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

        # end
        losses = torch.zeros(50)
        for i in range(50):
            x, y = get_batch(val_data)
            _, losses[i] = self(x, y)
        print(
            f"epoch {epoch}, loss {loss.item():.2f}, val loss {losses.mean().item():.2f}"
        )


if __name__ == "__main__":
    # get text
    text = get_text()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # character encoding and decoding
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda x: [stoi[ch] for ch in x]
    itos = {i: ch for i, ch in enumerate(chars)}
    decode = lambda x: "".join([itos[i] for i in x])

    # load data
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # 90% train, 10% val, no shuffle
    train_data, val_data = data[:n], data[n:]

    model = Transformer().to(device)
    model.train(train_data, val_data, iters=n_iters, lr=lr)

    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(ctx, max_tokens=out_tokens)[0].tolist()))
