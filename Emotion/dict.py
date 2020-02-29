import numpy as np
import torch


word2index = np.load('/Users/amos/Documents/Python/Emotion/word2index2.npy',allow_pickle=True).item()
index2word = np.load('/Users/amos/Documents/Python/Emotion/index2word2.npy',allow_pickle=True).item()
word2count = np.load('/Users/amos/Documents/Python/Emotion/word2count2.npy',allow_pickle=True).item()
n_words = len(word2count)
#为避免字典中找不到字的情况，wordToTensor中会处理
#同时，在index2word中也需处理
index2word[n_words] = 'UN'
n_words += 1


def wordToTensor(word):
    tensor = torch.zeros(1,n_words)
    #避免字典中找不到数字的情况
    if word in word2index:
        tensor[0][word2index[word]]=1
    else:
        tensor[0][n_words-1] = 1
    return tensor
def textToTensor(line):
    tensor = torch.zeros(len(line),1,n_words)
    for li,word in enumerate(line):
        tensor[li][0] = wordToTensor(word)
    return tensor
