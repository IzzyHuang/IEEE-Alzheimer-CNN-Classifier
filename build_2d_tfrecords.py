import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import pandas as pd

from tensorflow import keras
from dltk.io import preprocessing
from matplotlib import pyplot as plt

import os

#CONSTANTS

# basic RAW databases, with registrated and skull-stripped images
DB_SS_PATH = "D:\ADNI\PRE"

# data subfolders (labels)
CLASS_SUBFOLDERS = ['CN/', 'MCI/', 'AD/']

# label mapping
LABELS = {'CN': 0, 'MCI': 1, 'AD': 2}

# shape of the images, both 3D and 2D
IMG_SHAPE = (78, 110, 86)
IMG_2D_SHAPE = (IMG_SHAPE[1] * 4, IMG_SHAPE[2] * 4)

# 2D supervised TFRecords database
DB_TF_2D_PATH = "D:\ADNI\TF"
TFREC_2D_SS_TRAIN = "TRAIN"
TFREC_2D_SS_TEST = "TEST"
TFREC_2D_SS_VAL = "VAL"

train_tfrec2D = os.path.join(DB_TF_2D_PATH, TFREC_2D_SS_TRAIN)
test_tfrec2D = os.path.join(DB_TF_2D_PATH, TFREC_2D_SS_TEST)
val_tfrec2D = os.path.join(DB_TF_2D_PATH, TFREC_2D_SS_VAL)

training_set, test_set, validation_set = None,None,None

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
  
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def split_data():
    """
    Split the data manually into training, testing and validation since tf records doesn't allow splitting later
    """
    TEST_SPLIT = 0.15
    VALIDATION_SPLIT = 0.15
    filenames = np.array([])

    # iterate all three class folders in the db
    for subf in CLASS_SUBFOLDERS:
    # using the skull stripped data
        path = DB_SS_PATH + subf
        for name in os.listdir(path):
            complete_name = os.path.join(path, name)
            if os.path.isfile(complete_name):
                filenames = np.concatenate((filenames, complete_name), axis=None)
    
    for i in range(1000):
        np.random.shuffle(filenames)
        
    test_margin = int(len(filenames) * TEST_SPLIT)
    training_set, test_set = filenames[test_margin:], filenames[:test_margin]

    validation_margin = int(len(training_set) * VALIDATION_SPLIT)
    training_set, validation_set = training_set[validation_margin:], training_set[:validation_margin]

    print('Training set:', training_set.shape)
    print('Validation set:', validation_set.shape)
    print('Test set:', test_set.shape)


def slices_matrix_2D(img):
  ''' Transform a 3D MRI image into a 2D image, by obtaining 9 slices 
      and placing them in a 4x4 two-dimensional grid.
      
      All 16 cuts are from a horizontal/axial view. They are selected
      from the 30th to the 60th level of the original 3D image.
      
      Parameters:
        img -- np.ndarray with the 3D image
        
      Returns:
        np.ndarray -- The resulting 2D image
  '''
  
  # create the final 2D image 
  image_2D = np.empty(IMG_2D_SHAPE)
  
  # set the limits and the step
  TOP = 60
  BOTTOM = 30
  STEP = 2
  N_CUTS = 16
  
  # iterator for the cuts
  cut_it = TOP
  # iterator for the rows of the 2D final image
  row_it = 0
  # iterator for the columns of the 2D final image
  col_it = 0

  for cutting_time in range(N_CUTS):
    
    # cut
    cut = img[cut_it, :, :]
    cut_it -= STEP
    
    # reset the row iterator and move the
    # col iterator when needed
    if cutting_time in [4, 8, 12]:
      row_it = 0
      col_it += cut.shape[1]
    
    # copy the cut to the 2D image
    for i in range(cut.shape[0]):
      for j in range(cut.shape[1]):
        image_2D[i + row_it, j + col_it] = cut[i, j]
    row_it += cut.shape[0]
  
  # return the final 2D image, with 3 channels
  # this is necessary for working with most pre-trained nets
  return np.repeat(image_2D[None, ...], 3, axis=0).T
  #return image_2D


def load_image_2D(abs_path, labels):
  ''' Load an image (.nii) and its label, from its absolute path.
      Transform it into a 2D image, by obtaining 16 slices and placing them
      in a 4x4 two-dimensional grid.
      
      Parameters:
        abs_path -- Absolute path, filename included
        labels -- Label mapper
        
      Returns:
        img -- The .nii image, converted into a numpy array
        label -- The label of the image (from argument 'labels')
        
  '''
  
  # obtain the label from the path (it is the last directory name)
  label = labels[abs_path.split('/')[-2]]
  
  # load the image with SimpleITK
  sitk_image = sitk.ReadImage(abs_path)
  
  # transform into a numpy array
  img = sitk.GetArrayFromImage(sitk_image)
  
  # apply whitening
  img = preprocessing.whitening(img)
  
  # make the 2D image
  img = slices_matrix_2D(img)
  
  return img, label

def create_tf_record_2D(img_filenames, tf_rec_filename, labels):
  ''' Create a TFRecord file, including the information
      of the specified images, after converting them into 
      a 2D grid.
      
      Parameters:
        img_filenames -- Array with the path to every
                         image that is going to be included
                         in the TFRecords file.
        tf_rec_filename -- Name of the TFRecords file.
        labels -- Label mapper
  '''
  
  # open the file
  writer = tf.python_io.TFRecordWriter(tf_rec_filename)
  
  # iterate through all .nii files
  for meta_data in img_filenames:

    # load the image and label
    img, label = load_image_2D(meta_data, labels)

    # create a feature
    feature = {'label': _int64_feature(label),
               'image': _float_feature(img.ravel())}

    # create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # serialize to string and write on the file
    writer.write(example.SerializeToString())
    
  writer.close()

def main():
  split_data()
  create_tf_record_2D(training_set, train_tfrec2D, LABELS)
  create_tf_record_2D(test_set, test_tfrec2D, LABELS)
  create_tf_record_2D(validation_set, val_tfrec2D, LABELS)


if __name__ == "__main__":
   main()

