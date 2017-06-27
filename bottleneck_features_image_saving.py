# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:14:56 2017

@author: Heinz
"""


# -----------------------------------------------------------------------------
# Importing required libraries
# -----------------------------------------------------------------------------


import os
import numpy as np
from skimage import data, io, filters, transform
from skimage.transform import resize
import pandas as pd
import shutil
import matplotlib
import matplotlib.pyplot as plt
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


# -----------------------------------------------------------------------------
# Set parameters
# -----------------------------------------------------------------------------


workdir = "d:/Projects/kaggle/InvasiveSpecies"

# Parameters for the transformations in Data Augmentation
idg_width_shift_range = 0.2
idg_height_shift_range = 0.2
idg_shear_range = 0.2
idg_zoom_range = 0.2
idg_horflip = True

# Parameters for data output (image size, number of trafos)
target_size = (224, 224)
num_versions = 3 # How many versions of each picture to generate
input_shape = target_size + (3, )
test_num_images = 40000 # Do set to large value to process all images

# Fraction of data for validation
val_fraction = 0.5

# Parameters for training
batch_size = 20

# threshold for classification
threshold_class = 0.5


# -----------------------------------------------------------------------------
# Read image labels
# -----------------------------------------------------------------------------


# Set working directory
os.chdir(workdir)


# Read labels of the training data
train_labels = pd.read_csv("data/train_labels.csv")


# I am getting weird error messages sometimes when running this cell. Don't know why.

data_root_dir = "data/vgg_impregen_bottleneck/"

subdirlist = ["train/0/", "train/1/",
              "val/0/", "val/1/",
              "test_0", "test_1"]


dirlist = []
for subdir in subdirlist:
    dirlist.append(data_root_dir + subdir)


# The test folders are not really test data. Instead, they keep a mirror of the validation
# data. The pictures will then be classified once again by our network and depending on
# correctness of result put in a subfolder. This somewhat convoluted scheme is necessary
# because of the syntax and limitation of keras.

for element in dirlist:
    if os.path.exists(element):
        shutil.rmtree(element)

for element in dirlist:
    os.makedirs(element)

    
# Prepare two image data generators. 
no_transform_datagen = ImageDataGenerator(rescale = 1./255)

image_prep_gen = ImageDataGenerator(rescale = 1./255,
                                   width_shift_range = idg_width_shift_range,
                                   height_shift_range = idg_height_shift_range,
                                   shear_range = idg_shear_range,
                                   zoom_range = idg_zoom_range,
                                   horizontal_flip = idg_horflip)
    
    

# Make augmented versions of every image, also scale down untransformed. Should
# help to use all images for training etc. 
for file_counter in range(0, min(train_labels.shape[0], test_num_images)):
    file_id = str(train_labels["name"][file_counter])

    # Load image, make it an array so the flow()-method can work with it.
    img = image_utils.load_img("data/train/" + file_id + ".jpg", target_size = target_size)
    img = image_utils.img_to_array(img)
    img = img.reshape((1, ) + img.shape)
    
    file_prefix = file_id.rjust(4, "0") + "_" + str(train_labels["invasive"][file_counter]) + "_"
    
    # Process each image twice
    
    # First do the image without transformation (for validation)
    save_to_dir = data_root_dir + "val/" + str(train_labels["invasive"][file_counter]) + "/"
    
    for batch in no_transform_datagen.flow(x = img, batch_size = 1, save_to_dir = save_to_dir,
                                            save_prefix = file_prefix, save_format = "jpg"):
        break
    
    # Second augment images
    save_to_dir = data_root_dir + "train/" + str(train_labels["invasive"][file_counter]) + "/"
    counter = 0
    for batch in image_prep_gen.flow(x = img, batch_size = 1, save_to_dir = save_to_dir,
                                            save_prefix = file_prefix, save_format = "jpg"):
        counter += 1
        if counter > (num_versions - 1):
            break

# Get the number of images in each subfolder
num_images_train_0 = len(os.listdir(data_root_dir + "train/0"))
num_images_train_1 = len(os.listdir(data_root_dir + "train/1"))
num_images_val_0 = len(os.listdir(data_root_dir + "val/0"))
num_images_val_1 = len(os.listdir(data_root_dir + "val/1"))

images_to_train = num_images_train_0 + num_images_train_1        
images_to_val = num_images_val_0 + num_images_val_1        

# Import VGG16 model
imagenet_model = applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

# In this generator, the batch size is set to 1 to process one image at a time
# The number of images to process is then determined by the steps argument in 
# predict generator
generator_for_bottleneck_train = no_transform_datagen.flow_from_directory(data_root_dir + "train",
                                                                   target_size = target_size,
                                                                   batch_size = 1,
                                                                   class_mode = None,
                                                                   shuffle = False)

startzeit = time.time()
bottleneck_train = imagenet_model.predict_generator(generator_for_bottleneck_train, images_to_train)
endzeit = time.time()

print("Duration of processing " + str(images_to_train) + " images was " + str(round(endzeit - startzeit, 2)))



generator_for_bottleneck_val = no_transform_datagen.flow_from_directory(data_root_dir + "val",
                                                                   target_size = target_size,
                                                                   batch_size = 1,
                                                                   class_mode = None,
                                                                   shuffle = False)

startzeit = time.time()
bottleneck_val = imagenet_model.predict_generator(generator_for_bottleneck_val, images_to_val)
endzeit = time.time()

print("Duration of processing " + str(images_to_val) + " images was " + str(round(endzeit - startzeit, 2)))

# Save arrays of predictions alongside labels
np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_train)
np.save(open('bottleneck_features_train_labels.npy', 'wb'), generator_for_bottleneck_train.classes)
np.save(open('bottleneck_features_val.npy', 'wb'), bottleneck_val)
np.save(open('bottleneck_features_val_labels.npy', 'wb'), generator_for_bottleneck_val.classes)

