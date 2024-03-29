import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

### Load the Training File and the filepath
from Create_Image_CSVs import df as train
dirname = os.path.dirname(__file__)

def Predict_Photo():

    ### Load the Opening Question
    ans = input('What photo would you like analyzed? Please insert the filepath:')

    ### Load the Model
    model = load_model(dirname + '/Model.h5')

    ### Predict a couple samples
    # Process the Images
    img = image.load_img(ans,target_size=(400,400,3))
    img = image.img_to_array(img)
    img = img/255
    # Predict
    proba = model.predict(img.reshape(1,400,400,3))

    if proba[0,0] > .9:
        print("That's someone else")
    if proba[0,1] > .9:
        print("Yo, That's Stephen A. Smith")
    if proba[0,0] <.9 and proba[0,1] <.9:
        print("I'm not sure about that one, but there's a", proba[0,0], "chance that is someone else and a", proba[0,1], "chance that's Stephen A.")


### Create Introduction to Running File
if __name__ == '__main__':
    print('Welcome to Yo, Is that Stephen A?')
    Predict_Photo()
    
