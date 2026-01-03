import torch
import torch.nn as nn 
import math
import torch.nn.functional as F

class LayerNorm(nn.Module): 
  def __init__(self, embed_dim): 
    super().__init__()
    self.scale = nn.Parameter(torch.ones(embed_dim)) 
    self.shift = nn.Parameter(torch.zeros(embed_dim))

  def forward(self, input):
    return F.layer_norm(input, self.scale.shape, self.scale, self.shift, 1e-5)

class GPT2(nn.Module): 

  def __init__(self, **kwargs): 
    super().__init__()

    vocab_size = kwargs["vocab_size"] 
    embed_dim = kwargs["embed_dim"] 
    block_size = kwargs["block_size"] 
    n_layer = kwargs["n_layer"]

    self.vocab_size = vocab_size

    # encoding the input 
    self.token_encoder = nn.Embedding(vocab_size, embed_dim)
    self.position_encoder = nn.Embedding(block_size, embed_dim)
    self.blocks = nn.ModuleList([
      self.AttentionBlock(embed_dim, block_size=1024, **kwargs["attention_block"]) for _ in range(n_layer)]
                                )
    self.final_layernorm = LayerNorm(embed_dim) 
    self.final_linearmap = nn.Linear(embed_dim, vocab_size)
    self.block_size = block_size

  class AttentionBlock(nn.Module): 
    """
    Basically MultiHeadAttention with MLP
    """
    def __init__(self, embed_dim: int, block_size: int, **kwargs): 
      super().__init__()
      self.ln_1 = LayerNorm(embed_dim)
      self.attn = self.MultiHeadAttention(embed_dim, n_head=kwargs["n_head"], block_size=block_size)
      self.ln_2 = LayerNorm(embed_dim)
      self.mlp = self.MLP(embed_dim, latent_dim_multiplier=4)

    class MultiHeadAttention(nn.Module): 

      mask: torch.Tensor

      def __init__(self, embed_dim: int, n_head: int, block_size: int): 
        super().__init__()
        self.map_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        self.n_head = n_head
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.embed_dim = embed_dim

      def forward(self, x): 
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (embed_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.map_qkv(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = self.output_projection(y)
        return y

    class MLP(nn.Module): 

      def __init__(self, embed_dim, latent_dim_multiplier): 
        super().__init__()
        self.c_fc    = nn.Linear(embed_dim, latent_dim_multiplier * embed_dim, bias=True)
        self.gelu    = nn.ReLU()
        self.c_proj  = nn.Linear(latent_dim_multiplier * embed_dim, embed_dim, bias=True)

      def forward(self, x): 
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

    def forward(self, x):
      x = x + self.attn(self.ln_1(x))
      x = x + self.mlp(self.ln_2(x))
      return x

  def forward(self, x: torch.Tensor): 
    # x is a tensor of shape B, T, where B is batch and T is length of sequence 
    _, T = x.size()
    
    token_embedding = self.token_encoder(x) 
    position_embedding = self.position_encoder(torch.arange(T).to(self.device)) # type: ignore
    x = token_embedding + position_embedding

    for block in self.blocks: 
      x = block(x) 
    x = self.final_layernorm(x) 
    logits = self.final_linearmap(x) 
    return logits

  def generate(self, x, max_tokens):
    """
    Take a conditioning sequence of indices x (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_tokens):
      # if the sequence context is growing too long we must crop it at block_size
      idx_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size:]
      # forward the model to get the logits for the index in the sequence
      logits = self(idx_cond)
      # pluck the logits at the final step and scale by desired temperature
      logits = logits[:, -1, :]
      # apply softmax to convert logits to (normalized) probabilities
      probs = F.softmax(logits, dim=-1)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1)
      # append sampled index to the running sequence and continue
      x = torch.cat((x, idx_next), dim=1)

    return x



