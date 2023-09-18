#importing required libraries
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from keras.applications.vgg19 import preprocess_input
import os
import pandas as pd
import numpy as np

#required to perform global average pooling
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten
#load the vgg-19 base model without the fully connected layers as we don't classify the images here
base_model = VGG19(weights='imagenet',include_top=False)

#code to make sure the trainable parameters are not accessible
for layer in base_model.layers:
    layer.trainable = False

#remove the max-pooling layer after block5_conv4 to use average pooling instead
x = base_model.layers[-2].output
x = GlobalAveragePooling2D()(x)

#create the model used to extract the features
model = Model(inputs=base_model.input, outputs=x)

#summary for reference
model.summary()

#load the excel file using pandas dataframe
df = pd.read_excel("/Users/tysonpais/PycharmProjects/Lifelog-Activity-Detection/TempData/training_data.xlsx")

#create a column to save image embeddings and initiate it with a dummy value
df['image_embedding']="NA"

#load the image folder
root_folder="/Users/tysonpais/Documents/Uni Koblenz/ML Research Lab/Implementation/u1/NTCIR14_Images"
embedding=[]

#iterate through the dataframe to access the image path
for index, row in df.iterrows():
    #extract the image path
    file = root_folder + '/' + row['image_path']
    #check if the file is present
    if os.path.isfile(file):
        #load the image here and resize it
        img = image.load_img(file, target_size=(224, 224))
        #convert it into an array
        x = image.img_to_array(img)
        #change the dimension
        x = np.expand_dims(x, axis=0)
        #process the input using vgg19 function, the images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling
        x = preprocess_input(x)
        #extract the feature vector
        features = model.predict(x)
        embedding.append(features)
        df.at[index, 'image_embedding'] = np.squeeze(features)

#save the embedding in a numpy array to be used as inputs for nueral network
np.save("TempData/ImageEmbeddingVGG19.npy",np.squeeze(embedding))
columns=['attribute_top01', 'attribute_top02', 'attribute_top03', 'attribute_top04', 'attribute_top05', 'attribute_top06', 'attribute_top07', 'attribute_top08', 'attribute_top09', 'attribute_top10', 'category_top01', 'category_top01_score', 'category_top02', 'category_top02_score', 'category_top03', 'category_top03_score', 'category_top04', 'category_top04_score', 'category_top05', 'category_top05_score', 'concept_class_top01', 'concept_score_top01', 'concept_bbox_top01', 'concept_class_top02', 'concept_score_top02', 'concept_bbox_top02', 'concept_class_top03', 'concept_score_top03', 'concept_bbox_top03', 'concept_class_top04', 'concept_score_top04', 'concept_bbox_top04', 'concept_class_top05', 'concept_score_top05', 'concept_bbox_top05', 'concept_class_top06', 'concept_score_top06', 'concept_bbox_top06', 'concept_class_top07', 'concept_score_top07', 'concept_bbox_top07', 'concept_class_top08', 'concept_score_top08', 'concept_bbox_top08', 'concept_class_top09', 'concept_score_top09', 'concept_bbox_top09', 'concept_class_top10', 'concept_score_top10', 'concept_bbox_top10', 'concept_class_top11', 'concept_score_top11', 'concept_bbox_top11', 'concept_class_top12', 'concept_score_top12', 'concept_bbox_top12', 'concept_class_top13', 'concept_score_top13', 'concept_bbox_top13', 'concept_class_top14', 'concept_score_top14', 'concept_bbox_top14', 'concept_class_top15', 'concept_score_top15', 'concept_bbox_top15', 'concept_class_top16', 'concept_score_top16', 'concept_bbox_top16', 'concept_class_top17', 'concept_score_top17', 'concept_bbox_top17', 'concept_class_top18', 'concept_score_top18', 'concept_bbox_top18', 'concept_class_top19', 'concept_score_top19', 'concept_bbox_top19', 'concept_class_top20', 'concept_score_top20', 'concept_bbox_top20', 'concept_class_top21', 'concept_score_top21', 'concept_bbox_top21', 'concept_class_top22', 'concept_score_top22', 'concept_bbox_top22', 'concept_class_top23', 'concept_score_top23', 'concept_bbox_top23', 'concept_class_top24', 'concept_score_top24', 'concept_bbox_top24', 'concept_class_top25', 'concept_score_top25', 'concept_bbox_top25', 'ADL', 'ENV', 'ADL_ENV']

#remove extra columns to avoid storage error
df = df.drop(columns=columns)

#save the dataframe for reference only, this file is not used in further steps
df.to_excel("TempData/outputVGG19lsx", index = False)