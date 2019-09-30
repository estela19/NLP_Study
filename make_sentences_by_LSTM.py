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
maxlen = 20
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
  sentences.append(text[i : i + maxlen])
  next_chars.append(text[i + maxlen])

#"""
print("sentences: ", sentences)
print("next_chars: ", next_chars)

print('학습할 구문의 수:', len(sentences))
print('텍스트를 ID 벡터로 변환합니다...')
x_one_hot = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y_one_hot = np.zeros((len(sentences), len(chars)), dtype = np.bool)

#print("x: ", x)
#print("y: ", y)

for i, sentence in enumerate(sentences):
  for t, char in enumerate(sentence):
    x_one_hot[i, t, char_indices[char]] = 1
  y_one_hot[i, char_indices[next_chars[i]]] = 1
  
#print("x_one_hot: ", x)
#print("y_one_hot: ", y)
#"""
  
#hyper parameters
dic_size = len(char_indices)
input_size = len(char_indices)
hidden_size = len(char_indices)
learning_rate = 1e-4
epochs = 100
batch_size = 64
"""
x_one_hot = []
for word in (sentences):
  x_idx = [char_indices[c] for c in word]
  one_hot = [np.eye(dic_size)[x] for x in x_idx]
  x_one_hot.append(one_hot)
  

y_idx = [[char_indices[c] for c in next_chars]]
y_one_hot = [np.eye(dic_size)[y] for y in y_idx]
"""
x_one_hot = torch.FloatTensor(x_one_hot)
y_one_hot = torch.LongTensor(y_one_hot)
#print(one_hot)
#print(x_one_hot)
#print (y_one_hot)
#print(x_one_hot.size())

#모델 생성
rnn = nn.LSTM(input_size, hidden_size, batch_first = True)
print("make LSTM")

#Loss func 와 optimizer 정의
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)
print("define ok")

#start training
for i in range(epochs):
  outputs, _status = rnn(x_one_hot)
#  print("processing..")
  loss = loss_func(outputs.contiguous().view(9954419, -1), y_one_hot.contiguous().view(-1))
#  print("loss ok")
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  if(i % 1 == 0):
    result = outputs.data.numpy().argmax(axis = 2)
    result_str = ''.join(chars[c] for c in np.squeeze(result).reshape(-1))
    print(i, "loss: ", loss.item(), "prediction str: ", result_str)

    
""" result
풀무맹맹맹들맹맹무풀얄맹맹반맹무
풀무맹맹맹들맹맹프맹둘냈둘반맹무
풀무맹맹맹들맹맹프맹둘맹둘꿰건얄풀풀무맹맹맹들맹맹프맹둘맹둘꿰얄풀얐무끈풀맹맹들맹맹프맹둘맹둘꿰얄풀얐맹둘룬락얄들록맹프맹둘맹둘꿰얄풀얐맹둘룬맹둘해냈훈패맹둘맹둘꿰얄풀얐맹둘룬맹둘해점맹된냈둘록둘꿰얄풀얐맹둘룬맹둘해점맹된록맹뺌만꿰얄풀얐맹둘룬맹둘해점맹된록맹뺌맹맹맹합얄얄둘룬맹둘해점맹된록맹뺌맹맹맹급맹둘깠낮얄둘해점맹된록맹뺌맹맹맹급맹둘된무둘락웬풀맹훈록맹뺌맹맹맹급맹둘된무둘무된맹냈훈록맹풀맹맹맹급맹둘된무둘무된맹얄맹뭣합풀얄맹맹급맹둘된무둘무된맹얄맹뭣급맹둘냈둘급맹둘된무둘무된맹얄맹뭣급맹둘된패맹냈둘풀무둘무된맹얄맹뭣급맹둘된패맹둘맹둘냈얄무된맹얄맹딩급맹둘된패맹둘맹둘된패맹패풀얄맹뭣급맹둘된패맹둘맹둘된패맹얄둘맹냈딩급맹둘된패맹둘맹둘된패맹얄둘맹맹얐얐냈둘풀패맹둘맹둘된패맹얄둘맹맹얐얐둘찼맹냈얘둘맹둘된돝맹얄둘맹맹얐얐둘찼맹앉맹무냈둘된풀맹얄들맹맹얐얐둘찼맹앉맹무둘프둘건얄얄들맹맹얐얐둘찼맹맹맹무둘프둘돝얄둘들얄맹뺌얐둘찼맹앉맹무둘프둘돝얄둘맹둘맹합얄둘찼맹앉맹무둘프둘돝얄둘맹둘맹맹얐맹냈얘앉훈무둘프둘돝얄둘맹둘맹맹얐맹된패돝합얄둘프둘돝얄둘맹둘맹맹얐맹된패돝깠돝맹냈둘랭얄둘맹둘맹맹얐맹된패돝깠돝맹해찼무냈만맹둘맹맹얐맹된패돝깠돝맹해찼무둘돝얐앉얄맹들맹된패돝깠돝맹해찼무둘돝얐맹둘맹락얄된패돝웬돝맹해찼무둘돝얐맹둘맹된맹맹른돝깠풀맹해찼무둘돝얐맹둘맹된맹맹둘맹냈건얄앉찼무둘돝얐맹둘맹된맹맹둘맹냈맹맹얐른무둘돝얐맹둘맹된맹맹둘맹냈맹맹얐맹둘맹합얄얄둘맹된맹맹둘맹냈맹맹얐맹둘맹둘맹훈실얄된맹맹둘맹냈맹맹얐맹둘맹둘맹훈뭣맹둘건둘둘맹냈맹맹얐맹둘맹둘맹훈뭣맹둘맹
둘셔록둘맹얐맹둘맹둘맹훈뭣맹둘맹
둘패맹얄합얄얄둘맹둘맹훈뭣맹둘맹
둘패맹얄둘뺌급실얄둘맹훈뭣맹둘맹
둘패맹얄둘뺌급무맹맹냈둘찼맹둘맹

감감감감감감!찬딴.'''''''( 
감감감감감감!놓딴.'''''''( 
감감감감감감!놓딴.'''''''( 
감감감감감감!놓딴.'''''''(
"""
