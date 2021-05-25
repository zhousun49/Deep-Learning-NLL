import torch
from torch.autograd import Variable
batch_size = 4
max_length = 3
hidden_size = 2
n_layers =1
feature_dim = 1

# container
batch_in = torch.zeros((batch_size, max_length, feature_dim))

# data
vec_1 = torch.FloatTensor([[1], [2], [3]])
vec_2 = torch.FloatTensor([[1], [2], [0]])
vec_3 = torch.FloatTensor([[1], [0], [0]])
vec_4 = torch.FloatTensor([[2], [0], [0]])

batch_in[0] = vec_1
batch_in[1] = vec_2
batch_in[2] = vec_3
batch_in[3] = vec_4
# print(batch_in[0])

batch_in = Variable(batch_in)
print(batch_in.size())
seq_lengths = [3,2,1,1] # list of integers holding information about the batch size at each sequence step

# pack it
pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths, batch_first=True)
print(pack)

rnn = torch.nn.RNN(feature_dim, hidden_size, n_layers, batch_first=True) 
h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))

#forward 
out, _ = rnn(pack, h0)

# unpack
unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
print(unpacked)
print(unpacked_len)