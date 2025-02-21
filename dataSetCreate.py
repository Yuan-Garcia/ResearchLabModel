import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchtext.vocab
import inflect
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import loadDataSet  # Assuming you have this custom module for loading datasets

p = inflect.engine()

# Generate dataset
data = []
for num in range(10000):
    word = p.number_to_words(num).replace(",", "").replace("-", " ")
    data.append((word, str(num)))

# Save dataset
df = pd.DataFrame(data, columns=["text", "number"])
df.to_csv("numbers_dataset.csv", index=False)

import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Extract texts (input) and nums (target)
    texts = [text for text, _ in batch]
    nums = [num for _, num in batch]
    
    # Pad the texts to ensure they are the same length (padding with zeros)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)  # You can adjust padding_value if needed
    
    # Pad the nums (if necessary)
    padded_nums = pad_sequence(nums, batch_first=True, padding_value=-1)  # You can adjust padding_value if needed

    return padded_texts, padded_nums

# Assuming NumberDataset is properly defined in loadDataSet
dataset = loadDataSet.NumberDataset(data)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Define your Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, char_vocab_size)

    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.lstm(x)
        x = self.decoder(x)
        return x


model = Seq2Seq(len(dataset.vocab), len(dataset.char_vocab))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Padding value used in 'num'
ignore_index = -1  # Assuming -1 is the padding value

# Define the loss function with ignore_index for padding
criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

for epoch in range(130, 200):  # Train for 10 epochs
    for text, num in dataloader:
        optimizer.zero_grad()

        # Pad 'num' to ensure equal length sequences in a batch
        num = rnn_utils.pad_sequence(num, batch_first=True, padding_value=ignore_index)

        # Forward pass
        output = model(text)

        # Check the shapes before reshaping
        print(f"Output shape before reshaping: {output.shape}")
        print(f"Num shape before reshaping: {num.shape}")

        # Ensure the output and num tensors are reshaped correctly: (batch_size * seq_len, num_classes)
        output = output.view(-1, len(dataset.char_vocab))  # Flatten to (batch_size * seq_len, num_classes)
        
        # Flatten 'num' to (batch_size * seq_len)
        num = num.view(-1)  # Flatten to (batch_size * seq_len)

        # Check the shapes after reshaping
        print(f"Output shape after reshaping: {output.shape}")
        print(f"Num shape after reshaping: {num.shape}")

        # Ensure the shapes match before calculating loss
        if output.shape[0] != num.shape[0]:
            print(f"Mismatch in batch sizes: output has {output.shape[0]}, num has {num.shape[0]}")
            continue  # Skip this batch if there is a mismatch

        # Calculate loss
        loss = criterion(output, num)  # CrossEntropyLoss expects both input and target to have same batch size
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")





# Prediction function
def predict(text):
    tokens = torch.tensor([dataset.vocab[token] for token in text.split()]).unsqueeze(0)
    output = model(tokens)
    predicted_num = "".join([dataset.char_vocab.get_itos()[idx] for idx in output.argmax(dim=-1).squeeze()])
    return predicted_num

# Test the prediction function
print(predict("one thousand four hundred and thirty two"))  # Expected output: "1432"
