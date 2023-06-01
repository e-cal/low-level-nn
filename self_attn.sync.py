# %% [markdown]
# # Self-Attention

# %%
import torch
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
