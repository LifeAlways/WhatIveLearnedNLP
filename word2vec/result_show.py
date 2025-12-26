from gensim.models import Word2Vec
model = Word2Vec.load("models/word2vec.model")

print(model.wv.most_similar("good"))
print(model.wv.most_similar("film"))
print(model.wv.most_similar("bad"))

print(model.wv.similarity("good", "great"))
print(model.wv.similarity("good", "bad"))
