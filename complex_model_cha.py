import tiktoken
import torch 
import torch.nn as nn
import os
from torch.nn import functional as F
#from nns import BigramLanguageModel
# download data

# read in data
with open('shakespear_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



# character tokenize
# get a list of non-repeating characters in the text, sorted (list(set(text))) --> volcabulary size
volcab = sorted(list(set(text)))
# tokenize the text -> character to number (pairs!)
volcab_to_num = {}
num_to_volcab = {}
for i, volcabulary in enumerate(volcab):
    volcab_to_num[volcabulary] = i
    num_to_volcab[i] = volcabulary

# Define encoder & decoder
    # e.g. "hii there" => [46, 47, 47, 1, 58, 46, 43, 56, 43] & can do the reverse
    # other (sentencePiece - subword unit)
    # GPT2 = tiktoken library 
def encode(sentence): return [volcab_to_num[c] for c in sentence]
def decode(num_list): return ''.join(num_to_volcab[i] for i in num_list)
# Tokenize the entire text -> pyTorch (tensor)
data = torch.tensor(encode(text), dtype=torch.long)

# Split into train, validate sets -> 90%, 10%
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Hyperparameter
# Block
    # Each time train on a subset of the entire text
    # When block_size is 8, it means we train 8 instances of prediction at once, so need to pass in 9 characters
    # x = [:block_size]; y(target) = [1:block_size + 1] -> train 1 charactor to block_size charactors
# Batch
    # batch_size = # of independent seqences(block_size) we process in parallel

block_size = 256
batch_size = 64
max_new_tokens = 100
learning_rate = 3e-4
volcab_length = len(volcab)
max_iteration = 5000
eval_iters = 200
eval_interval = 500
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

# use torch.manual_seed to get random slice
torch.manual_seed(1337)
# get a batch from the given dataset
    # get a list of random sampling location (number = batch_size)
    # sample x and y(target) -> tensor of tokens
def get_batch(data_set_name):
    data_set = train_data if data_set_name == "train" else val_data
    sample_loc = torch.randint(len(data_set) - block_size, (batch_size,))
    x = []
    y = []
    for i in sample_loc:
        x.append(data_set[i : i+block_size])
        y.append(data_set[i+1 : i+block_size+1])
    x = torch.stack(x)
    y = torch.stack(y)
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

n_embed = 384
n_layer = 6
n_head = 6
dropout = 0.1

class OneHeadSelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size))) # lower triangular matrix
        self.dropout = nn.Dropout(dropout)
        
    # attention (Q, K, V) = softmax(QK.T /sqrt(head_size))V
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x) # (B, T, C)
        
        # compute attention score (affinities)
        weight = q @ k.transpose(-2, -1) * C ** -0.5 # QK.T /sqrt(head_size), (B, T, C) @ (B, C, T) --> (B, T, T)
        weight_filtered = weight.masked_fill(self.tril[: T, : T] == 0, float('-inf')) # (B, T, T)
        weight_filtered = F.softmax(weight_filtered, dim=-1)
        weight_filtered = self.dropout(weight_filtered)
        
        result = weight_filtered @ v #(B, T, T) @ (B, T, C) --> (B, T, C)
        return result

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([OneHeadSelfAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) # linear projection of the result. 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        result = torch.cat([h(x) for h in self.heads], dim=-1)
        result = self.dropout(self.proj(result))
        return result
        
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed,n_embed), # projection
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadSelfAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.ln_f = nn.LayerNorm(n_embed) # final layer
        #self.sa_head = MultiHeadSelfAttention(4, n_embed//4)
        #self.feed_forward = FeedForward(n_embed)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embd + pos_embd # (B,T,C)
        #x = self.sa_head(x)
        #x = self.feed_forward(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # makesure size is <= block_size
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



model_file_name = "shakespear_complex"
# Train
m = BigramLanguageModel(volcab_length)
m = m.to(device)
if os.path.isfile(model_file_name):
    m.load_state_dict(torch.load(model_file_name))
    print("Model loaded.")
else:
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), learning_rate)
    for steps in range(max_iteration): 
        # every once in a while evaluate the loss on train and val sets
        if steps  % eval_interval == 0 or steps  == max_iteration - 1:
            losses = estimate_loss()
            print(f"step {steps }: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

initial_input = torch.zeros((1, 1), dtype=torch.long,device=device)
output = m.generate(initial_input, max_new_tokens)[0].tolist()
print(decode(output))
torch.save(m.state_dict(), model_file_name)