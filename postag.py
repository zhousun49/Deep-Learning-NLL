import sys
import torch 
import ast
from lstmtagger import LSTMTagger
from inputdata import InputData
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import numpy as np
import math

##Assign inputs
inputs = sys.argv
file_to_tag = inputs[1]
outputs = inputs[2]

##Open dictionary file
f = open("dict.txt", "r")
token = ast.literal_eval(f.readline())
tag = ast.literal_eval(f.readline())

##Transver Key Value
transversed = {}
for key, value in tag.items():
    transversed[value] = key 
# print(transversed)

##Sentence file and transfer to token
token_file = open(file_to_tag, encoding="utf-8")
sentences = []
sentence_length = []
for i in token_file:
    single_line = i.rstrip('\n').split(" ")
    sentences.append(single_line)
    sentence_length.append(len(single_line))
train_token = np.ones((len(sentence_length),100),dtype = int)
dummy_POS_token = np.ones((len(sentence_length),100),dtype = int)
for i in range(len(sentence_length)):
    one_sent = sentences[i]
    for one_word in one_sent: 
        ind = one_sent.index(one_word)
        if one_word.lower() in token:
            train_token[i][ind] = token[one_word.lower()]
    for j in range(sentence_length[i], 100):
        train_token[i][j] = 0
# for j in train_token:
#     print(j)

##Load in to DataLoader
train_mat = InputData(train_token, dummy_POS_token)
train_loader = DataLoader(dataset=train_mat, batch_size=1383, sampler=SequentialSampler(train_mat))
# print(len(sentence_length))
training_length_mat = np.array_split(sentence_length, math.ceil(len(sentence_length)/100))
# print(training_length_mat)

##Load model
emb_size = 100
hidden_size = 100
model = LSTMTagger(emb_size, hidden_size, len(token), len(tag))
model.load_state_dict(torch.load('./model.pt'))
model.eval()

print("Length of sentences: ", len(sentences))
##Prediction
# print(sentences[300])
# scores = torch.zeros(math.ceil(len(sentence_length)/100), 100, 100)
for (batch_num, batch) in enumerate(train_loader):
    tag_scores = model(batch[0], sentence_length) 
    print(tag_scores.size())
    max_scores = torch.argmax(tag_scores, dim=2)
    # scores[batch_num] = max_scores
    # print(batch_num)
    print(max_scores.size())
    for j in range(len(max_scores)):
        for k in range(len(sentences[j+batch_num*100])):
            single_value = max_scores[j][k].item()
            # print(single_value)
            # print(sentences[j][k])
            if single_value != 0:
                # print("sentences: ", sentences[j*100][k])
                # print("transverse: ", transversed[single_value])
                sentences[j+batch_num*100][k] = sentences[j+batch_num*100][k] + '/' + transversed[single_value]

joined = []
##Join List
for j in sentences:
    joined.append(" ".join(j))

##Write to File
with open(outputs, 'w') as f:
    for item in joined:
        f.write("%s\n" % item)
