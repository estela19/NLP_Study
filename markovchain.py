import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Twitterimport urlib.request

import os, re, json, random

#마르코프 체인 딕셔너리 만들기
def make_dic(words):
  tmp = ["@"]
  dic = {}
  for word in words:
    tmp.append(word)
    if len(tmp) < 3: continue
    if len(tmp) > 3: tmp = tmp[1:]
    set_word3(dic, tmp)
    
    if word == ".":
      tmp = ["@"]
      continue
  return dic

#딕셔너리에 데이터 등록하기
def set_word3(dic, s3):
  w1, w2, w3 = s3
  if not w1 in dic: dic[w1] = {}
  if not w2 in dic[w1]: dic[w1][w2] = {}
  if not w3 in dic[w1][w2]: dic[w1][w2][w3] = 0
  dic[w1][w2][w3] += 1
  
#문장 만들기
def make_sentence(dic):
  ret = []
  if not "@" in dic: return "no dic"
  top = dic["@"]
  w1 = word_choice(top)
  w2 = word_choice(top[w1])
  ret.append(w1)
  ret.append(w2)
  while True:
    w3 = word_choice(dic[w1][w2])
    ret.append(w3)
    if w3 == ".": break
    w1, w2, = w2, w3
  ret = "".join(ret)
  return ret

def word_choice(sel):
  deys = sel.keys()
  return random.choice(list(keys))

#문장 읽어 들이기
document_file = "heungbuga.txt"
dict_file = "markov_heungbuga.json"
if not os.path.exists(dict_file):
  #토지 텍스트 파일 읽어 들이기
  fp = codecs.open("BEXX0003.txt", "r", encoding = "utf-16")
  soup = BeautifulSoup(fp, "html.parser")
  body = soup.select_one("body > text")
  text = body.getText()
  #형태소 분석
  twitter = Twitter()
  malist = twitter.pos(text, norm = True)
  words = []
  for word in malist:
    #.을 제외한 구두점은 제외
    if not word[1] in ["Punctuation"]:
      words.append(word[0])
    if word[0] == ".":
      words.append(word[0])
      
  #딕셔너리 생성
  dic = make_dic(words)
  json.dump(dic, open(dict_file, "w", encoding = "utf-8"))
  
else:
  dic = json.load(open(dict_file, "r"))
  
  
#문장생성
for i in range(3):
  s = make_sentence(dic)
  print(s)
  print("---")
