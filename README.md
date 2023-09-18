# Lifelog-Activity-Detection
THIS IS A COPY OF A PRIVATE REPOSITORY. CONTRIBUTIONS HAVE BEEN MADE BY THREE COLLABORATORS. (SARAH SAJID, TYSON PAIS, SHINJINI SHOME - Students of University of Koblenz under the supervision of PROF. DR. FRANK HOPFGARTNER)  
WE USE NTCIR-14 DATA TO RUN THE LIFELOG ACTIVITY DETECTION TASK TO CLASSIFY IMAGES TO 16 PREDEFINED LABELS.  
PLEASE NOTE: THERE ARE NO DATASETS HERE AS IT IS SENSITIVE INFORMATION, ALL DATA AND FILES THAT CAN BE SENSITIVE HAVE BEEN REMOVED.

## Dataset Organization
#### 1. Dataset folder 
Contains visual concepts and annotations (To be downloaded from source)


<img width="324" alt="image" src="https://user-images.githubusercontent.com/96388629/215270571-e2e186cd-933e-495d-8135-72f9f49e49ec.png">

#### 2. Temp Data
Contains intermediate processed files used for assigning 16 official classes to the data



## Data Preprocessing
#### 1. image_filtering.py
Images with high blurriness and homogeneity are ignored. The Threshold for blurriness is set to 10 and homogeneity is set to 90%.
Some examples of rejected images are:

<img width="200" alt="image" src="https://user-images.githubusercontent.com/11594284/219966166-4b36e484-52bc-4b3b-bda0-bb27518e57a5.JPG"><img width="200" alt="image" src="https://user-images.githubusercontent.com/11594284/219966197-b988d5ae-de3c-48a3-a30b-4bb13a270c8b.JPG">

#### 2. Calibration
As the dataset images are captured using OMG Autographer camera that has a 136° field of view (fov), a fisheye effect is produced. This fisheye effect is corrected using Adobe lightroom. In the reference research, the purpose for this step is to get more accurate visual features from computer vision models. Before and after comparison of images is as follows:

<img width="821" alt="image" src="https://user-images.githubusercontent.com/96388629/220445444-50bcfcc6-6c2f-443e-a88c-f142bc436ab8.png">

<img width="701" alt="image" src="https://user-images.githubusercontent.com/96388629/220445743-6d9cfcf5-0615-4f1c-97fb-7fe6721c4f3b.png">



## Manual Annotation
Each image is assigned an activity class. The official NTCIR classes are 16. This assignment is done manually based on activity and environment attributes in the provided data.

#### 1. assign_official_classes.py
After having the above mentioned data in place, the first step is to run this file. It assigns the 16 official classes to all the data (currently for u1)

#### 2. sync_images_and_data.py
The next step is to run this file. Make sure you edit the image path. The result is a training_data.xlsx file which contains the training data only for available images. The data for which images are not available is discarded.



## Word Embeddings
#### 1. create_vocabulary.py
Creates a vocabulary of all the words that we want to encode and later use in training (attributes, categories, concepts)

#### 2. encode_vocabulary.py
Encodes the vocabulary using word2vec



## Visual Features
#### 1. image_embeddingsVGG19.py
Creates image embeddings using VGG-19. A 512D vector is produced and is saved in ImageEmbeddingVGG19.npy file, which is later used as input feature for DNN. It uses training_data.xlsx produced from sync_images_and_data.py execution to maintain the order of image embeddings and the encoded labels.

#### 2. image_embeddingsVGG16.py
Creates image embeddings using VGG-16. A 512D vector is produced and is saved in ImageEmbeddingVGG16.npy file, which is later used as input feature for DNN. It uses training_data.xlsx produced from sync_images_and_data.py execution to maintain the order of image embeddings and the encoded labels.

## Label Encoding
#### 1. label_encoding.py
Inorder to feed the labels(official classes) to the Nueral Network, they have to be encoded first. This file uses MultiLabelBinarizer to encode the labels. All the 13 classes are encoded in 1s or 0s and stored in LabelEncoding.npy.  It uses training_data.xlsx produced from sync_images_and_data.py execution to maintain the order of image embeddings and the encoded labels.

## Basic DNN Model
#### 1. baselineNN.py
This is the implementation of a Deep Nueral Network dicussed in the research paper that only uses visual fetaures as input. 
Overview of the model:
1. Uses files ImageEmbeddingVGG19.npy/ImageEmbeddingVGG19.npy and LabelEncoding.npy as input
2. Learning rate is set to 0.00001
3. Uses Adam optimzer
4. Uses sigmoid activation in the output layer(13 output nuerons)
5. Uses binary cross entropy as loss function
6. Uses Earlystopping mechanism to prevent overfitting
7. 40 epochs for execution
8. Uses precision, recall and micro f1-score for model evaluation with threshold of 0.5
