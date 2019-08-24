from gensim.models import word2vec

#데이터 불러오기
data = word2vec.Text8Corpus("./drive/My Drive/textfiles/wiki.wakati")
#모델생성
model = word2vec.Word2Vec(data, size = 100)
#모델저장
model.save("wiki.model")
print("ok")

#test
model.most_similar(positive = ["서울", "맛집", "음식"])
