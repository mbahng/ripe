import torch
from torch.utils.data import Dataset
import os

def tokenize_text(text_path):
  """Tokenize text and build vocabulary."""
  with open(text_path, 'r') as f:
    text = f.read()

  # Build vocabulary
  chars = sorted(set(text))
  vocab_size = len(chars)

  # Create mappings
  stoi = {ch: i for i, ch in enumerate(chars)}
  itos = {i: ch for i, ch in enumerate(chars)}

  # Tokenize
  tokens = [stoi[c] for c in text]

  return tokens, vocab_size, stoi, itos

class TextDataset(Dataset):
  def __init__(self, data, block_size, vocab_size, stoi, itos):
    self.data = data
    self.block_size = block_size
    self.vocab_size = vocab_size 
    self.stoi = stoi 
    self.itos = itos

  def __len__(self):
    return len(self.data) - self.block_size

  def __getitem__(self, idx):
    x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
    y = torch.tensor(self.data[idx + 1:idx + 1 + self.block_size], dtype=torch.long)
    return x, y

def shakespeare(dataset_cfg: dict):
  block_size = dataset_cfg["block_size"] 
  if not os.path.exists("data/shakespeare.txt"): 
    os.makedirs("data", exist_ok=True) 
    import requests
    url = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
    response = requests.get(url)
    if response.status_code == 200:
      with open("data/shakespeare.txt", "w") as file:
        file.write(response.text)
    else: 
      print("Could not retrieve shakespeare from online.")
  tokens, vocab_size, stoi, itos = tokenize_text("data/shakespeare.txt")

  # Split
  train_split, val_split, test_split = dataset_cfg["split"]
  total_split = train_split + val_split + test_split
  train_split, val_split, test_split = train_split / total_split, val_split / total_split, test_split / test_split
  
  train_end_idx = int(len(tokens) * train_split) 
  val_end_idx = int(len(tokens) * (train_split + val_split))

  train_tokens = tokens[:train_end_idx]
  val_tokens = tokens[train_end_idx:val_end_idx]
  test_tokens = tokens[val_end_idx:]

  # Create datasets
  train_ds = TextDataset(train_tokens, block_size, vocab_size, stoi, itos)
  val_ds = TextDataset(val_tokens, block_size, vocab_size, stoi, itos)
  test_ds = TextDataset(test_tokens, block_size, vocab_size, stoi, itos)

  return train_ds, val_ds, test_ds
