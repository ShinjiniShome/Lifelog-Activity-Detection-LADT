import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
learning_rate = 0.00001
optimizer = Adam(learning_rate=learning_rate)

image_embedding=np.load('/Users/tysonpais/Documents/Projects/Lifelog-Activity-Detection/TempData/ImageEmbeddingVGG16.npy')
word_embedding=np.load('/Users/tysonpais/PycharmProjects/Lifelog-Activity-Detection/WordEmbeddings/WordEmbeddingsWord2Vec.npy')


label_encoding=np.load('/Users/tysonpais/Documents/Projects/Lifelog-Activity-Detection/TempData/LabelEncoding.npy')

#model for image embedding before concatenation 512 to 128
base_model = Sequential()
base_model.add(Dense(256, input_dim=512, activation='relu'))
base_model.add(Dense(128, activation='relu'))
base_model.add(Dropout(0.5))
base_model.summary()

visual_feature = base_model.predict(image_embedding)
concatenated_embeddings=np.hstack((visual_feature,word_embedding))

#model post concatenation
model = Sequential()
model.add(Dense(13, activation='sigmoid'))



X=visual_feature
y=label_encoding
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# compile the model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[])
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1, mode='auto')
# train the model
model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1,callbacks=[early_stop])

y_pred = model.predict(X_test)
print(y_pred)
y_pred_binary = (y_pred > 0.5).astype(int)
print(y_pred_binary)
precision = precision_score(y_test, y_pred_binary, average='micro')
recall = recall_score(y_test, y_pred_binary, average='micro')
f1 = f1_score(y_test, y_pred_binary, average='micro')
print('Precision: {:.3f}, Recall: {:.3f}, F1-score: {:.3f}'.format(precision, recall, f1))

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
# plot the loss curves
plt.plot(train_loss, label='train')
plt.plot(val_loss, label='test')
plt.legend()
plt.show()
