import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Twitter
from gensim.models import word2vec

#원본파일(html파일)불러오기 및 파싱
fp = codecs.open("./drive/My Drive/textfiles/4BH20002.txt", "r", encoding = "utf-16")
soup = BeautifulSoup(fp, "html.parser")
body = soup.select_one("body > text")
text = body.getText()

#형태소 분석
twitter = Twitter()
results = []
lines = text.split("\n")
for line in lines:
  malist = twitter.pos(line, norm = True, stem = True)
  r = []
  for word in malist:
  #조사, 어미, 구두점은 분석에서 제외
    if not word[1] in ["Josa", "Eomi", "Punctuation"]:
      r.append(word[0])
      
  rl = (" ".join(r)).strip()
  results.append(rl)
#  print(rl)

#파일로 출력
wakati_file = 'heungbuga.wakati'
with open(wakati_file, 'w', encoding = 'utf-8') as fp:
  fp.write("\n".join(results))
  

#word2vec 모델 생성
data = word2vec.LineSentence(wakati_file)
model = word2vec.Word2Vec(data, size = 200, window = 10, hs = 1, min_count = 2, sg = 1)
model.save("./drive/My Drive/heungbuga.model")
print("ok")

#모델 불러오기 및 테스트
model = word2vec.Word2Vec.load("./drive/My Drive/heungbuga.model")
model.most_similar(positive = ["흥보"])
