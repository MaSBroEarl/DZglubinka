import torch
import torch.nn as nn
from torchtext.datasets import CoNLL2000Chunking
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def yield_tokens_and_tags(data_iter, tokenizer, tag_type="pos"):
    for sentence in data_iter:
        tokens, pos_tags, _ = zip(*sentence)
        if tag_type == "pos":
            yield tokens, pos_tags

tokenizer = get_tokenizer("basic_english")

def build_vocab(data_iter, tokenizer, tag_type="pos"):
    token_vocab = Counter()
    tag_vocab = Counter()
    for tokens, tags in yield_tokens_and_tags(data_iter, tokenizer, tag_type):
        token_vocab.update(tokens)
        tag_vocab.update(tags)
    return build_vocab_from_iterator([token_vocab.keys()], specials=["<pad>", "<unk>"]), build_vocab_from_iterator([tag_vocab.keys()], specials=["<pad>"])

class POSDataset(torch.utils.data.Dataset):
    def __init__(self, data_iter, token_vocab, tag_vocab, tokenizer, tag_type="pos"):
        self.samples = []
        for tokens, tags in yield_tokens_and_tags(data_iter, tokenizer, tag_type):
            token_ids = [token_vocab[token] for token in tokens]
            tag_ids = [tag_vocab[tag] for tag in tags]
            self.samples.append((torch.tensor(token_ids), torch.tensor(tag_ids)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    tokens, tags = zip(*batch)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=token_vocab["<pad>"])
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=tag_vocab["<pad>"])
    return tokens_padded, tags_padded


class LSTMPOSModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embed_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=token_vocab["<pad>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

train_iter = CoNLL2000Chunking(split='train')
token_vocab, tag_vocab = build_vocab(train_iter, tokenizer)

train_iter = CoNLL2000Chunking(split='train')  # Перезагрузить итератор, так как он был исчерпан
train_dataset = POSDataset(train_iter, token_vocab, tag_vocab, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

model = LSTMPOSModel(len(token_vocab), len(tag_vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tag_vocab["<pad>"])
optimizer = torch.optim.Adam(model.parameters())


epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for tokens, tags in train_loader:
        tokens, tags = tokens.to(device), tags.to(device)
        optimizer.zero_grad()
        outputs = model(tokens)
        loss = criterion(outputs.view(-1, len(tag_vocab)), tags.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

print("Training complete!")