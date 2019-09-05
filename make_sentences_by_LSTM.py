import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms

import codecs
import numpy as np
from bs4 import BeautifulSoup
import random, sys

#"""
fp = codecs.open("./drive/My Drive/textfiles/4BH20002.txt", "r", encoding = "utf-16")
soup = BeautifulSoup(fp, "html.parser")
body = soup.select_one("body > text")
text = body.getText() + " "
print('Corpus length : ', len(text))
#"""


#문자 읽어 들이고 ID 붙이기
chars = sorted(list(set(text)))
print('사용되고 있는 문자의 수:', len(chars))
#문자 -> ID
char_indices = dict((c, i) for i, c in enumerate(chars))
#ID -> 문자
indices_char = dict((i, c) for i, c in enumerate(chars))

print("char_indices: ", char_indices)

#텍스트를 maxlen개의 문자로 자르고 다음에 오는 문자 등록
maxlen = 3
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
  sentences.append(text[i : i + maxlen])
  next_chars.append(text[i + maxlen])

"""
print("sentences: ", sentences)
print("next_chars: ", next_chars)

print('학습할 구문의 수:', len(sentences))
print('텍스트를 ID 벡터로 변환합니다...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)), dtype = np.bool)

#print("x: ", x)
#print("y: ", y)

for i, sentence in enumerate(sentences):
  for t, char in enumerate(sentence):
    x[i, t, char_indices[char]] = 1
  y[i, char_indices[next_chars[i]]] = 1
  
print("x: ", x)
print("y: ", y)
"""
  
#hyper parameters
dic_size = len(char_indices)
input_size = len(char_indices)
hidden_size = len(char_indices)
learning_rate = 1e-3
epochs = 100
batch_size = 64

x_one_hot = []
for word in (sentences):
  x_idx = [char_indices[c] for c in word]
  one_hot = [np.eye(dic_size)[x] for x in x_idx]
  x_one_hot.append(one_hot)
  

y_idx = [[char_indices[c] for c in next_chars]]
y_one_hot = [np.eye(dic_size)[y] for y in y_idx]

x_one_hot = torch.FloatTensor(x_one_hot)
y_one_hot = torch.LongTensor(y_one_hot)
#print(one_hot)
print(x_one_hot)
#print (y_one_hot)
#print(x_one_hot.size())

#모델 생성
rnn = nn.RNN(input_size, hidden_size, batch_first = True)

#Loss func 와 optimizer 정의
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)

#start training
for i in range(epochs):
  outputs, _status = rnn(x_one_hot)
  loss = loss_func(outputs.contiguous().view(36, -1), y_one_hot.contiguous().view(-1))
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
#  if(i % 10 == 0):
#    result = outputs.data.numpy().argmax(axis = 2)
#    result_str = ''.join([chars[c] for c in np.squeeze(result)])
#    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)
print(i, "loss: ", loss.item())
