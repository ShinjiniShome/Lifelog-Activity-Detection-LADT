from gensim.models import KeyedVectors

#Pre-Trained Word2Vec Model
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

result1 = model.most_similar(positive=['woman', 'doctor'], negative=['man'], topn=3)
result2 = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=3)
result3 = model.most_similar(positive=['man', 'doctor'], negative=['woman'], topn=3)
result4 = model.most_similar(positive=['man', 'queen'], negative=['woman'], topn=3)
result5 = model.most_similar(positive=['man', 'king'], negative=['woman'], topn=3)
result6 = model.most_similar(positive=['woman', 'teacher'], negative=['man'], topn=3)
result7 = model.most_similar(positive=['woman', 'plumber'], negative=['man'], topn=3)
result8 = model.most_similar(positive=['woman', 'carpenter'], negative=['man'], topn=3)

print(result1)
print(result2)
print(result3)
print(result4)
print(result5)
print(result6)
print(result7)
print(result8)

print("\n\n\n\n")

#Pre-Trained GloVe Model
from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'glove.6B.300d.txt'
model = KeyedVectors.load_word2vec_format(filename, binary=False, no_header=True)

result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)