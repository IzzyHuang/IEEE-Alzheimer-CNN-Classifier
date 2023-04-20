import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import pandas as pd

from tensorflow import keras
from dltk.io import preprocessing
from matplotlib import pyplot as plt

import os

import image_preprocess

#CONSTANTS

IMG_SHAPE = (78, 110, 86)
IMG_2D_SHAPE = (IMG_SHAPE[1] * 4, IMG_SHAPE[2] * 4)
IMG_RGB_SHAPE = (IMG_SHAPE[1] * 4, 
                 IMG_SHAPE[2] * 4, 
                 3)
ATLAS = "/home/avs/ie3_ml/mn305_atlas.nii"
MODEL_FILE = "/home/avs/ie3_ml/model/trained_cnn.h5"
REG_DB = "/home/avs/ie3_ml/client/reg"
SKULL_STRIPPED_DB = "/home/avs/ie3_ml/client/ss"

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
  #label = labels[abs_path.split('/')[-2]]
  
  # load the image with SimpleITK
  sitk_image = sitk.ReadImage(abs_path)
  
  # transform into a numpy array
  img = sitk.GetArrayFromImage(sitk_image)
  
  # apply whitening
  img = preprocessing.whitening(img)
  
  # make the 2D image
  img = slices_matrix_2D(img)
  
  return img

def get_prediction(img_dir, img_name):
    process_class = image_preprocess.ImagePreprocess(ATLAS, True)
    path = process_class.register_and_save(img_dir,img_name)
    new_path = os.path.join(SKULL_STRIPPED_DB,img_name)
    process_class.skull_strip_nii(path, new_path, frac=0.2)
    new_path += ".gz"
    image_2d = load_image_2D(new_path,[])

    image_2d = np.reshape(image_2d, (1,440,344,3))

    model = tf.keras.models.load_model(MODEL_FILE)
    
    os.remove(path)
    os.remove(new_path)

    # Make a prediction using your model
    predicted_probs = model.predict(image_2d)
    CN = predicted_probs[0][0]*100
    AD = predicted_probs[0][1]*100
    pred = {
      "CN": "{:0.2f}".format(CN),
      "AD": "{:0.2f}".format(AD)
    }
    return pred


def main(img_dir, img_name):
    
    predicted_probs = get_prediction(img_dir, img_name)
    
    with open(os.path.join(img_dir, "preds.txt"),"a" ) as fhand:
      fhand.write(f"{img_name}: {predicted_probs}\n")
    # Print the predicted probabilities
    #print(predicted_probs)


if __name__ == "__main__":
  img_dir = "/home/avs/ie3_ml/test_images/"
  for img in ["CN", "AD"]:
    img_name = img + ".nii"
    main(img_dir, img_name)
  