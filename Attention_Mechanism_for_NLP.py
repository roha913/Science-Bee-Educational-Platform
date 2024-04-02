import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Assuming we have a vocabulary size for both the source and target languages
SRC_VOCAB_SIZE = 1000  # Source language vocabulary size
TGT_VOCAB_SIZE = 1000  # Target language vocabulary size
EMBED_SIZE = 256  # Embedding size
HIDDEN_SIZE = 512  # Hidden size of the RNN/LSTM
BATCH_SIZE = 32

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        
    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = torch.sum(self.v * torch.tanh(self.attention(torch.cat((h, encoder_outputs), 2))), 2)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, -1, embed_size)
        attn_weights = self.attention(hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.gru(rnn_input, hidden)
        output = self.out(output.squeeze(0))
        return output, hidden, attn_weights

# Example usage
encoder = Encoder(SRC_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(TGT_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)

# Assuming src and tgt are batched sequences of word indices for source and target languages
src = torch.randint(0, SRC_VOCAB_SIZE, (10, BATCH_SIZE))  # Example source sequence
tgt = torch.randint(0, TGT_VOCAB_SIZE, (10, BATCH_SIZE))  # Example target sequence

encoder_outputs, hidden = encoder(src)
outputs, hidden, attn_weights = decoder(tgt[0], hidden, encoder_outputs)

print(outputs.shape)  # Shape: [BATCH_SIZE, TGT_VOCAB_SIZE]
