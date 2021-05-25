import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.utils.rnn as rnnutils

class LSTMTagger(nn.Module):
    def __init__(self, emb_size, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_size = emb_size
        self.embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        # print("Vocab Size: ", vocab_size)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = emb_size, hidden_size = hidden_dim, num_layers = 1, bidirectional=True, batch_first = True, bias=False)
        self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)

    def forward(self, sentence, sent_length):
        X = self.embeddings(sentence)
        X = torch.nn.utils.rnn.pack_padded_sequence(X, sent_length, batch_first=True, enforce_sorted=False)
        lstm_out, hidden = self.lstm(X)
        unp,_ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first = True, total_length = 100)
        tag_space = self.hidden2tag(unp)
        # print(tag_space)
        # print("After linear Size: ", tag_space.size())
        # tag_scores = F.log_softmax(tag_space, dim=1)
        # print("Tag Score Dimension: ", tag_scores.size())
        return tag_space