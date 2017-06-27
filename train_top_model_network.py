# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 22:23:31 2017

@author: Heinz
"""


import os
import numpy as np
import pandas as pd
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
from keras import backend as K
from keras import applications
from keras import optimizers
import warnings
warnings.filterwarnings('ignore')

# Load functions to read images
from keras.preprocessing import image as image_utils
import time


target_size = (224, 224)
input_shape = target_size + (3, )


class_threshold = 0.5

data_root_dir = "data/vgg_impregen_bottleneck/"

workdir = "d:/Projects/kaggle/InvasiveSpecies"

os.chdir(workdir)



train_data = np.load(open("bottleneck_features_train.npy", "br"))
train_labels = np.load(open("bottleneck_features_train_labels.npy", "rb"))
val_data = np.load(open("bottleneck_features_val.npy", "rb"))
val_labels = np.load(open("bottleneck_features_val_labels.npy", "rb"))


top_model = Sequential()
top_model.add(Flatten(input_shape = train_data.shape[1:]))
top_model.add(Dense(256, activation = "relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(128, activation = "relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation = "sigmoid"))

top_model.compile(optimizer = "rmsprop", loss = "binary_crossentropy",
                  metrix = ["accuracy"])

# Takes a 2-5 minutes with 10 epochs
top_model.fit(train_data, train_labels,
              epochs = 20,
              batch_size = 30,
              validation_data = (val_data, val_labels))


val_predictions = top_model.predict(val_data, batch_size = 1)

# The output of the predict method is an array of shape (x, 1), needs to be
# recast into shape (x)
val_predictions.shape = len(val_predictions)

val_predictions_bin = val_predictions > class_threshold

successvek = val_predictions_bin == val_labels.astype(bool)

successvek = np.zeros(shape = len(val_predictions_bin))

for zaehler in range(0, len(successvek)):
    if val_predictions_bin[zaehler] and val_labels[zaehler].astype(bool):
        successvek[zaehler] = 1
    elif (not val_predictions_bin[zaehler]) and (not val_labels[zaehler].astype(bool)):
        successvek[zaehler] = 1

sum(successvek)

# To be super-sure that the predicted labels are linked to the correct picture,
# I'm gonna redo the whole prediction thing, this time with a image generator
# out of test_0 and test_1 directories.

# Make image generator 

imagenet_model = applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

# The compound model
model = Model(inputs = imagenet_model.input, outputs = top_model(imagenet_model.output))


notrafo_gen = ImageDataGenerator(rescale = 1./255)

val_generator = notrafo_gen.flow_from_directory(directory = data_root_dir + "val/",
                                                target_size = target_size,
                                                batch_size = 1,
                                                class_mode = None,
                                                shuffle = False)

# Run the complete model on the real image data (not pre-saved data)
startzeit = time.time()
neuval = model.predict_generator(val_generator, 2295)
endzeit = time.time()
print("Duration of processing " + str(1) + " images was " + str(round(endzeit - startzeit, 2)))


# Make paths in order to store images according to class and correctness of
# classification
def testMakePath (newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)

testMakePath(data_root_dir + "test_0/correct")
testMakePath(data_root_dir + "test_0/incorrect")
testMakePath(data_root_dir + "test_1/correct")
testMakePath(data_root_dir + "test_1/incorrect")        

# Use the classification threshold defined above. Go through all images, query
# its class and what the model thinks, then copy them into the appropriate
# folder.

for zaehler in range(0, len(neuval)):
    # Naming scheme comes in handy
    image_name = val_generator.filenames[zaehler]
    real_image_class = image_name[0]
    pred_image_class = neuval[zaehler] > class_threshold
    if real_image_class == "0":
        if pred_image_class == False:
            shutil.copy(src = data_root_dir + "val/" + image_name,
                        dst = data_root_dir + "test_0/correct")
        else:
            shutil.copy(src = data_root_dir + "val/" + image_name,
                        dst = data_root_dir + "test_0/incorrect")
    else:
        if pred_image_class == False:
            shutil.copy(src = data_root_dir + "val/" + image_name,
                        dst = data_root_dir + "test_1/incorrect")
        else:
            shutil.copy(src = data_root_dir + "val/" + image_name,
                        dst = data_root_dir + "test_1/correct")
        
    









