#import the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

#load the data
df = pd.read_excel("/Users/tysonpais/PycharmProjects/Lifelog-Activity-Detection/TempData/training_data.xlsx")
labels=[]

#pre-process the column
for x in df['Official_Classes']:
    labels.append(x.split(','))

without_spaces = [[word.strip() for word in lst] for lst in labels]

#use the library to convert
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(without_spaces)

#save the results in a numpy file to be used as input for NN in later stages
Encoded_labels = [sublist for sublist in encoded_labels]
np.save("TempData/LabelEncoding.npy",Encoded_labels)
df.to_excel("TempData/output.xlsx", index = False)