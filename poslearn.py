import sklearn
import torch
import torch.optim as optim
import torch.nn as nn
import sys
import numpy as np
from inputdata import InputData
from lstmtagger import LSTMTagger
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from sklearn.metrics import f1_score

#File reading
inputs = sys.argv
train_dir = inputs[1]
dev_dir = inputs[2]
train = open(train_dir, encoding="utf-8")
dev = open(dev_dir, encoding="utf-8")

#Assign words and POS list
#Both word_dict and POS_dict are in the correct order 
words = []
POS = []
training_sentences = []
training_length_mat = []
dev_sentences = []
dev_length_mat = []
for i in train:
    single_line = i.rstrip('\n').split(" ")
    training_sentences.append(single_line)
    training_length_mat.append(len(single_line))
    for j in single_line:
        sep = j.split("/")
        words.append(sep[0].lower())
        tag = sep[1]
        POS.append(tag)
for i in dev: 
    single_line = i.rstrip('\n').split(" ")
    dev_sentences.append(single_line)
    dev_length_mat.append(len(single_line))
unique, counts = np.unique(words, return_counts=True)
words_dict = dict(zip(unique, counts))
words_dict = dict(sorted(words_dict.items(), key=lambda item: item[1], reverse=True))
print('Length of total words dic:', len(words_dict))
unique, counts = np.unique(POS, return_counts=True)
POSs_dict = dict(zip(unique, counts))
POSs_dict = dict(sorted(POSs_dict.items(), key=lambda item: item[1], reverse=True))
print('Length of total POS dic:', len(POSs_dict))

#Assign real word dictionary with a value
word_dict = {}
POS_dict = {}
count = 2
word_dict['<PAD>'] = 0
word_dict['<UNK>'] = 1
POS_dict['<PAD>'] = 0
POS_dict['<UNK>'] = 1
for j in words_dict:
    if words_dict[j] > 1 and len(word_dict) < 1000:
        word_dict[j] = count
    else:
        break
    count += 1
count = 2
for j in POSs_dict:
    POS_dict[j] = count
    count += 1
print('training sentences: ', len(training_length_mat))
print('development sentences: ', len(dev_length_mat))
f = open("dict.txt","w")
f.write(str(word_dict))
f = open("dict.txt","a")
f.write('\n')
f.write(str(POS_dict))
print("Dictionary successfully saved as txt file")

#Assign arrays
train_token = np.zeros((len(training_length_mat),100),dtype = int)
POS_train_token = np.zeros((len(training_length_mat),100),dtype = int)
dev_token = np.zeros((len(dev_length_mat),100),dtype = int)
POS_dev_token = np.zeros((len(dev_length_mat),100),dtype = int)
for i in range(len(training_length_mat)):
    one_sent = training_sentences[i]
    for one_word in one_sent: 
        ind = one_sent.index(one_word)
        sep = one_word.split("/")
        if sep[0].lower() in word_dict:
            train_token[i][ind] = word_dict[sep[0].lower()]
        if sep[1] in POS_dict:
            POS_train_token[i][ind] = POS_dict[sep[1]]
    for j in range(training_length_mat[i], 100):
        train_token[i][j] = 0
        POS_train_token[i][j] = 0

f = open("vvvvv.txt","w")
for i in range(len(dev_length_mat)):
    one_sent = dev_sentences[i]
    sent = []
    for one_word in one_sent: 
        ind = one_sent.index(one_word)
        sep = one_word.split("/")
        sent.append(sep[0])
        if sep[0].lower() in word_dict:
            dev_token[i][ind] = word_dict[sep[0].lower()]
        if sep[1] in POS_dict:
            POS_dev_token[i][ind] = POS_dict[sep[1]]
    sent = " ".join(sent)
    f.write(sent)
    f.write("\n")
    for j in range(dev_length_mat[i], 100):
        dev_token[i][j] = 0
        POS_dev_token[i][j] = 0
print("Empty dev file saved")

##Loading into datasets and split training length matrix for batch training
train_mat = InputData(train_token, POS_train_token)
train_loader = DataLoader(dataset=train_mat, batch_size=100, sampler=SequentialSampler(train_mat))
training_length_mat = np.array_split(training_length_mat, 50)
dev_mat = InputData(dev_token, POS_dev_token)
dev_loader = DataLoader(dataset=dev_mat, batch_size=100, sampler=SequentialSampler(dev_mat))
dev_length_mat = np.array_split(dev_length_mat, 20)

##Define LSTM Model and Optimizer
emb_size = 100
input_size = emb_size
hidden_size = emb_size
model = LSTMTagger(emb_size, hidden_size, len(word_dict), len(POS_dict))
loss_function = nn.CrossEntropyLoss(ignore_index = 0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

f1_max = 0
for j in range(10):
##Run one epoch Training
    epoch_loss_train = 0
    model.train()
    for (batch_num, batch) in enumerate(train_loader):
        optimizer.zero_grad()
        tag_scores = model(batch[0], training_length_mat[batch_num]) 
        input_batch = batch[1].float()
        input_batch = input_batch.type(torch.LongTensor)
        reshapped_scores = torch.reshape(tag_scores, (tag_scores.shape[0]*tag_scores.shape[1],tag_scores.shape[2]))
        reshapped_input_batch = torch.reshape(input_batch, (-1,))
        loss = loss_function(reshapped_scores, reshapped_input_batch)
        loss.backward()
        optimizer.step()
        epoch_loss_train += loss.item()
    print("Average Loss in epoch " , j+1, " is: ", epoch_loss_train/50)

    ##Run one epoch validating
    epoch_loss_dev = 0
    model.eval()
    max_scores_lst = torch.zeros(20,10000)
    real_labels_lst = torch.zeros(20,10000)

    for (batch_num, batch) in enumerate(dev_loader):
        tag_scores = model(batch[0], dev_length_mat[batch_num]) 
        input_batch = batch[1].float()
        input_batch = input_batch.type(torch.LongTensor)
        max_scores = torch.argmax(tag_scores, dim=2)
        reshapped_scores = torch.reshape(max_scores, (-1,))
        reshapped_input_batch = torch.reshape(input_batch, (-1,))
        max_scores_lst[batch_num] = reshapped_scores
        real_labels_lst[batch_num] = reshapped_input_batch
        epoch_loss_dev += loss.item() 
    prediction = torch.reshape(max_scores_lst, (-1,))
    label = torch.reshape(real_labels_lst, (-1,))
    pred = []
    lab = []
    for i in range(len(label)): 
        if label[i].item() != 0 and label[i].item() != 1: 
            pred.append(prediction[i].item())
            lab.append(label[i].item())
    f1 =  f1_score(pred, lab, average = "macro")
    if f1 > f1_max:
        f1_max = f1
        torch.save(model.state_dict(), 'model.pt')
        print("F1 Score of epoch", j+1, " is: ", f1)
