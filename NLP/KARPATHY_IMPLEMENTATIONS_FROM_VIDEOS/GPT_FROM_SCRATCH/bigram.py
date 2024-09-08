""" TO DO: weight sharing, use ALIBI (or ROPE) pos_embs, use subword tokenization(tiktoken BPE) """

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange
import math

#hyperparameters
batch_size = 64
block_size = 256 # len of sequence (T(Time) dimension)
max_iters = 5000 # epochs
eval_interval = 500 # number of steps to evaluate 
learning_rate = 3e-4
device = 'cuda'
eval_iters = 200 # how many samples will be used for evaluation (from eval set) 
n_embd = 512 # embedding size
n_layer = 3 # transformer decoder layers
n_head = 16 # number of heads in multi head self attention
dropout = 0.2 # scale of dropout
# -------------

torch.manual_seed(42)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[i] for i in s]
decode = lambda s: ''.join([itos[i] for i in s])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of selt-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # attn scores
        wei = (q @ k.transpose(2, 1)) / (C ** 0.5) # (B, T, 16) @ (B, 16, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v # (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

# better than simple LeakyReLU
class GeLUThatWasUsedInGPT2(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4*n_embd)
        self.relu = GeLUThatWasUsedInGPT2() #self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(4*n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.linear2(self.relu(self.linear1(x))))

class Block(nn.Module):
    """ Transformer block (decoder without cross-attention) """
    def __init__(self, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffn = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # x + --> skip connection
        x = x + self.ffn(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embs = nn.Embedding(vocab_size, n_embd) 
        self.position_embs = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_head=n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embs = self.token_embs(idx) # Batch, Time (block_size), Channels(n_embd)
        pos_emb = self.position_embs(torch.arange(T, device=device)) # (T (block_size), C (n_embd))
        x = token_embs + pos_emb # B, T, C
        x = self.blocks(x)
        logits = self.lm_head(x) # Batch, Time (block_size), Channels(vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # for cross_entropy
            targets = targets.view(B*T) # for cross_entropy, the same: logits.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx - B, T
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # B, C (because we use last time step)
            probs = F.softmax(logits, dim=-1) # B, C
            # sampling
            idx_next = torch.multinomial(probs, num_samples=1) # B, 1
            # append sampled idx to the running sequence
            idx = torch.cat((idx, idx_next), dim=-1) # B, T+1
        return idx

model = BigramLanguageModel()
model = model.to(device)
print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in trange(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    opt.zero_grad()
    logits, loss = model(xb, yb)
    loss.backward()
    opt.step()

def main():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

if __name__ == '__main__':
    main()
