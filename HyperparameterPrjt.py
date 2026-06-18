# Necessary libraries
import os
import re
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import utils
import keras
import imageio 
from PIL import Image
# Reading the data
train = pd.read_csv('age_detection_train/train.csv')
# Image resizing of train data into single numpy array
temp = []
for img_name in train.ID:
    img_path = os.path.join('age_detection_train/Train', img_name)
    img = imageio.imread(img_path)
    img = np.array(Image.fromarray(img).resize((32, 32))).astype('float32')    
    temp.append(img)
train_x = np.stack(temp)
# Normalizing the images
train_x = train_x / 255.
# Encoding the categorical variable to numeric
lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = utils.np_utils.to_categorical(train_y)
# Specifying all the parameters we will be using in our network
input_num_units = (32, 32, 3)
hidden_num_units = 500
output_num_units = 3
epochs = 100
batch_size = 128
