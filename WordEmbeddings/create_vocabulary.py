import pandas as pd

visual_concepts = pd.read_excel("./TempData/training_data_with_classes.xlsx")
vocabulary = []

# columns attribute_top01 till attribute_top10
top10_attributes = visual_concepts.iloc[:, 3:13]

for col in top10_attributes:
    unique = top10_attributes[col].unique()
    for item in unique:
        items = str(item).split("/")
        for word in items:
            vocabulary.append(word)

# columns category_top01 till category_top05
for i in range(5):
    category = "category_top0" + str(i+1)
    unique = visual_concepts[category].unique()
    for item in unique:
        items = str(item).split("/")
        for word in items:
            vocabulary.append(word)

# columns concept_class_top01 and concept_class_top02, rest of the concept class fields are too sparse
for i in range(2):
    concept = "concept_class_top0" + str(i + 1)
    unique = visual_concepts[concept].unique()
    for item in unique:
        items = str(item).split("/")
        for word in items:
            vocabulary.append(word)

# remove leading and trailing white spaces, replace - and _ between words with space and drop duplicates
unique_vocabulary = pd.DataFrame(vocabulary).replace(r"^ +| +$", r"", regex=True).replace('-', ' ', regex=True).replace('_', ' ', regex=True).drop_duplicates()
unique_vocabulary.to_csv('./TempData/vocabulary.csv', index=False, header=None)

# print(list(unique_vocabulary[0]), '\n', len(list(unique_vocabulary[0])))

