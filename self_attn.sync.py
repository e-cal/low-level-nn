# %% [markdown]
# # Self-Attention
# Notes from Karpathy's lecture

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# %%
# batch, time, channels
B, T, C = 4, 8, 2
x = torch.rand(B, T, C)
x.shape

# %% [markdown]
# Simple averaging model (bag-of-words)
# Goal:
# $x[b,t] = \text{mean}_{\,i\, \le\, t} (x[b,i])$
#
# - up to 8 tokens in a batch
# - want them to "talk to each other"
#   - but only with tokens that came before them
#   - information flows from previous context to the current timestep
#   - no information from the future, trying to predict it
# - simplest way for a token to "communicate" with other tokens is to take the mean of all previous tokens
#   - creates a feature vector that "summarizes" the current token in the context of the previous tokens

# %%
xbow = torch.zeros((B, T, C))  # bow = bag-of-words
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]  # (t, C)
        xbow[b, t] = xprev.mean(dim=0)  # averages over time

# batch 0
print(x[0])
# NOTE: first row is the same since there is no previous context to average over
print(xbow[0])

# %% [markdown]
# Implementation above is very inefficient $O(n^2)$
# The "mathematical trick" is to use matrix multiplication
#
# - multiplying by a lower triangular ones matrix will sum all previous tokens
# - if instead of using ones, we use a uniform probability distribution (all rows sum to 1)
#       we get an average of all previous tokens
#   - weighted aggregation, where the weights are equal in the lower triangle

# %%
# multiplying by a lower triangular ones matrix
a = torch.tril(torch.ones(3, 3))
b = torch.randint(0, 10, (3, 2)).float()
print(f"a=\n{a}")
print(f"b=\n{b}")
print(f"a⋅b=\n{a @ b}")

# %%
# multiplying by a lower triangular uniform probability distribution
a = torch.tril(torch.ones(3, 3))
a = a / a.sum(dim=1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
print(f"a=\n{a}")
print(f"b=\n{b}")
print(f"a⋅b=\n{a @ b}")

# %% [markdown]
# Bag of words vectorized

# %%
w = torch.tril(torch.ones(T, T))
w = w / w.sum(dim=1, keepdim=True)
w

# %%
xbow2 = w @ x  # (T, T) ⋅ (B, T, C) -> (B, T, C) (pytorch adds the batch dimension)
xbow2[0]

# %%
# matrix multiplication essentially the same as the for loop above
xbow2.allclose(xbow)

# %% [markdown]
# Third implementation using softmax

# %%
tril = torch.tril(torch.ones(T, T))
w = torch.zeros((T, T))

# 1->0, 0->-inf
w = w.masked_fill(tril == 0, float("-inf"))

# exponentiate and divide by sum
# effectively normalizes so the row sums to 1 (gives the same as above)
w = F.softmax(w, dim=-1)

xbow3 = w @ x

# %%
torch.allclose(xbow, xbow3)

# %% [markdown]
# - When training self attention, weights start off as 0, giving the uniform distribution
# - As the model trains, the weights will change to reflect the importance of each previous token to the current token
# - Giving a weighted average instead of a uniform average
# - So, we use softmax to allow this training and get the weighted probability distributions

# %% [markdown]
"""
## Self-Attention Head

- builds on the ideas above
  - lower triangular weight matrix + softmax = weighted average of previous tokens
  - i.e. gives an affinity score to each previous token for the current token
- want to modify the zero weights to reflect the importance of each previous token
  - data dependent token associations



**Self attention solves this by having every node (every token at every timestep) emit 2 vectors:**
1. **(K)ey - what do I contain?**
2. **(Q)uery - what am I looking for?**

**Then, to get the affinity between tokens, take the dot products between the current token's query and every other token's key**

**This becomes the weights for the weighted average of the previous tokens**

"""


# %%
torch.manual_seed(1337)
B, T, C = 4, 8, 32  # 4×8 tokens with 32 channels of information per token
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)  # (B, T, head_size)
q = query(x)  # (B, T, head_size)

# only transpose the last two dimensions, leave batch dimension
w = q @ k.transpose(-2, -1)  # (B, T, head_size) ⋅ (B, head_size, T) -> (B, T, T)

tril = torch.tril(torch.ones(T, T))
w = w.masked_fill(tril == 0, float("-inf"))
w = F.softmax(w, dim=-1)

# aggregate the inputs before getting the output
# x is "information private to the current token"
# v is that information aggregated
v = value(x)  # (B, T, head_size)
out = w @ v

out.shape

# %%
w[0]

# %% [markdown]
"""
## Notes on attention

1. Attention is a **communication mechanism** between tokens
- tokens in a block can be thought of as nodes in a directed graph
- each node contains a vector of info
- can aggregate info via a weighted sum from all nodes that point to it 
    - (in a data dependent manner)
- a token's node is pointed to by all previous tokens in the block at time T + itself

2. There is no notion of (geometric) space
- attention acts on a set of vectors
- the nodes have no inherent position
- this is why we need positional encoding

3. Batches are completely independent
- no information persists between batches
- each batch is a completely independent graph

4. **Decoder vs Encoder blocks**
- only _decoder_ attention blocks prevent the current token from "communicating" with future tokens
- the triangular masking makes this a decoder attention block
- deleting that line allows all tokens to communicate with each other
    - i.e. an _encoder_ attention block

5. **Self-attention vs Cross-attention**
- self-attention means the keys, queries, and values all come from the same source
- can be generalized to cross attention, where the queries still come from $x$, but the keys and values come from a different source
    - other source may be encoder blocks encoding some context we want to condition on

6. **Scaled attention**
- divides weights $w$ by $\sqrt{\text{head_size}}$ 
- when $Q,K$ are unit variance, $w$ will have unit variance too
    - without scaling, $w$ will have variance ~head_size
- this keeps softmax diffuse and prevents saturation (some values way overpowering others)
    - saturated softmax approaches one hot encoding
"""

# %%
# scaled attention illustration

# no scaling
k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)
w = q @ k.transpose(-2, -1)
print("no scaling")
print(f"k.var())={k.var():.3f}")
print(f"q.var())={q.var():.3f}")
print(f"w.var())={w.var():.3f}")

# scaling
w_scaled = w * (head_size**-0.5)  # ≡ w / head_size^2
print("\nscaled")
print(f"w_scaled.var())={w_scaled.var():.3f}")

# %%
# saturated softmax illustration

vals = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
diffuse = torch.softmax(vals, dim=-1)
saturated = torch.softmax(vals * 8, dim=-1)

print(f"diffuse={diffuse}")
print(f"saturated={saturated}")
