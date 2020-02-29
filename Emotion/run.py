import pandas as pd
import numpy as np
import re
import random
import torch.nn as nn
import torch
from dict import n_words, textToTensor
from model import RNN
from model import hidden_size
from dataloader import train_data, train_label


rnn = RNN(n_words,hidden_size,3)
all_categories = [-1,0,1]

def categoryFromOutput(output):
    top_n,top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i],category_i


def randomTrainingExample():
    n = random.randint(1,len(train_data))
    text = train_data[n]
    if isinstance(text,float):
        # print(text)
        # print(n)
        # print(train_data[n])
        # print(train_label[n])
        n -= 1
        text = train_data[n]
    text_tensor = textToTensor(text)
    category = train_label[n]
    category_tensor = torch.tensor([all_categories.index(category)],dtype=torch.long)
    return category,text,category_tensor,text_tensor


criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.005)


def train(category_tensor, text_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()
    for i in range(text_tensor.size()[0]):
        output, hidden = rnn(text_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()


import time
import math

n_iters = 10000
print_every = 100
plot_every = 10

current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for iter in range(1, n_iters + 1):
    category, text, category_tensor, text_tensor = randomTrainingExample()
    output, loss = train(category_tensor, text_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print(
            '%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, text, guess, correct))
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

