import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms

import codecs
import numpy as np
from bs4 import BeautifulSoup
import random, sys

fp = codecs.open("./drive/My Drive/textfiles/4BH20002.txt", "r", encoding = "utf-16")
soup = BeautifulSoup(fp, "html.parser")
body = soup.select_one("body > text")
text = body.getText() + " "
print('Corpus length : ', len(text))

#문자 읽어 들이고 ID 붙이기
chars = sorted(list(set(text)))
print('사용되고 있는 문자의 수:', len(chars))
#문자 -> ID
char_indices = dict((c, i) for i, c in enumerate(chars))
#ID -> 문자
indices_char = dict((i, c) for i, c in enumerate(chars))

#텍스트를 maxlen개의 문자로 자르고 다음에 오는 문자 등록
maxlen = 20
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
  sentences.append(text[i : i + maxlen])
  next_chars.append(text[i + maxlen])
print('학습할 구문의 수:', len(sentences))
print('텍스트를 ID 벡터로 변환합니다...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)), dtype = np.bool)
for i, sentence in enumerate(sentences):
  for t, char in enumerate(sentence):
    x[i, t, char_indices[char]] = 1
  y[i, char_indices[next_chars[i]]] = 1
