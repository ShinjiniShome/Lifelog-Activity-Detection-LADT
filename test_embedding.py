# Load pre-trained Word2Vec model
import gensim.downloader as api
model = api.load('word2vec-google-news-300')

# Prepare visual concept
visual_concept = "cat"

# Encode visual concept as word embedding
embedding = model[visual_concept]
print(embedding)
