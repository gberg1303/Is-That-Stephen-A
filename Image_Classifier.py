import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


### Load the Dataframe of Files from the other Python Script
from Create_Image_CSVs import df as train
dirname = os.path.dirname(__file__)

### Start creating the train dataset
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img(os.path.join(dirname, 'Data/Training_Dataset/')+train['FileName'][i],target_size=(400,400,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

### Check that the files were loaded
plt.imshow(X[2])

### Drop the ID File and leave just the outcomes
y = np.array(train.drop(['FileName'],axis=1))
y.shape

### Randomly Select 10% of Images for a Validation Set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

### Create a CNN Model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

### Get the summary of the model
print(model.summary())

### Compile the Model 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

### Train the Model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

### Save Model
model.save('/Users/jonathangoldberg/Google Drive/Random/Random Fun/Is That Spehen A/Model.h5')

