import numpy as np
import pandas as pd
glove_all_embeddings=np.load("/Users/tysonpais/PycharmProjects/Lifelog-Activity-Detection/TempData/encoded_visual_concepts_glove.npy")
visual_concepts=pd.read_csv("/Users/tysonpais/PycharmProjects/Lifelog-Activity-Detection/TempData/vocabulary.csv",header=None)

encoding=dict(zip(visual_concepts[0],glove_all_embeddings))
print(encoding.keys())


def FindEncoding(word):
    if word in encoding:
        return np.average(encoding[word])


data=pd.read_excel("/Users/tysonpais/Documents/Projects/Lifelog-Activity-Detection/TempData/training_data.xlsx")
ndf = pd.DataFrame(data, columns=['attribute_top01', 'attribute_top02',
       'attribute_top03', 'attribute_top04', 'attribute_top05',
       'attribute_top06', 'attribute_top07', 'attribute_top08','attribute_top09', 'attribute_top10','category_top01','category_top02','category_top03','category_top04','category_top05','concept_class_top01','concept_class_top02'])

word_embed=[]
for row,column in ndf.iterrows():
    row_embed=[]
    for word in column.values:
        if word in encoding:
            row_embed.append(FindEncoding(word))
        else:
            row_embed.append(0)
    word_embed.append(row_embed)

print(word_embed)
np.save("WordEmbeddingsGlove.npy",word_embed)

