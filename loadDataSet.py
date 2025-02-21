import torch
from torch.utils.data import Dataset, DataLoader
# chatGPT told me about this import (build_vocab_from_iterator)
import torchtext
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# using the pytorch documentation, I found these useful imports

class NumberDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.vocab = torchtext.vocab.build_vocab_from_iterator([text.split() for text, _ in data], specials=["<pad>", "<sos>", "<eos>"])
        self.char_vocab = torchtext.vocab.build_vocab_from_iterator([[ch for ch in num] for _, num in data], specials=["<pad>", "<sos>", "<eos>"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, number = self.data[idx]
        text_tokens = [self.vocab[token] for token in text.split()]
        num_tokens = [self.char_vocab[ch] for ch in number]
        return torch.tensor(text_tokens), torch.tensor(num_tokens)
    

