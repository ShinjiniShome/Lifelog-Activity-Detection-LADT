import csv
import numpy as np
import gensim.downloader as api  # Load pre-trained Word2Vec model
#Use word2vec
model_w2v = api.load('word2vec-google-news-300')

#Use GloVe
model_glove = api.load('glove-wiki-gigaword-300')

# Load visual concepts from vocabulary CSV file
visual_concepts = []
with open('../TempData/vocabulary.csv', newline='') as vocabulary:
    reader = csv.reader(vocabulary)
    for row in reader:
        visual_concepts.append(row[0])

#Counters
#count1 =0
#count2 =0
#count3 =0
#count4 =0

# Encoding visual concepts with pre-trained word embeddings from Word2Vec
embeddings_w2v = []

for concept_w2v in visual_concepts:
    if concept_w2v in model_w2v:  # Encode only if the concept is present in the pre-trained model else it generates an error
        embedding_w2v = model_w2v[concept_w2v]
        embeddings_w2v.append(embedding_w2v)
        #count1 = count1 + 1
    else:
        embeddings_w2v.append(np.zeros((300,)))
        #count2 = count2 + 1


# Encoding visual concepts with pre-trained word embeddings from GloVe
embeddings_glove = []
for concept_glove in visual_concepts:
    if concept_glove in model_glove:  # Encode only if the concept is present in the pre-trained model else it generates an error
        embedding_glove = model_w2v[concept_glove]
        embeddings_glove.append(embedding_glove)
        #count3 = count3 + 1
    else:
        embeddings_glove.append(np.zeros((300,)))
        #count4 = count4 + 1


#print("Word2Vec...words present " + str(count1))
#print("Word2Vec...words absent " + str(count2))
#print("Glove...words present " + str(count3))
#print("Glove...words absent " + str(count4))
# Write encoded visual concepts to a CSV file
#np.savetxt("../TempData/encoded_visual_concepts.csv", embeddings, delimiter=",")

# Save encoded visual concepts from word2vec to NPY file
np.save("../TempData/encoded_visual_concepts_word2vec.npy", embeddings_w2v)

# Save encoded visual concepts from word2vec to NPY file
np.save("../TempData/encoded_visual_concepts_glove.npy", embeddings_glove)