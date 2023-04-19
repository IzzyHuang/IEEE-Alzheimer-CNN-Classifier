# NeuroNet - A Deep Learning Model for Alzheimer's Diagnosis 

## Introduction

This project is a deep learning model that can diagnose Alzheimer's Disease through MRI data. Moreover, it can even detect Alzheimer's early, in the form of early mild cognitive impairment and late mild cognitive impairment. While this model isn't meant to be a definitive test to diagnose Alzheimer's, it can serve as an initial diagnostic tool. 

The model is a convolutional neural network that analyzes 3D MRI images. A much more thorough explanation of the technical specifications, including image preprocessing and the model training is in the next few sections.

This project was inspired by the wonderful work conducted by Oscar Darias Plasencia that can be seen [here](https://towardsdatascience.com/alzheimer-diagnosis-with-deep-learning-a-survey-265406fa542a). 

P.S - the combination of medical imaging libraries and ML packages meant that the project in its entirety could not be run on any on system. Therefore, the files are littered with system specific constants and methods. Should you choose to play with the model, you could download the file from [here]() and look at the client file for how to interface with it. 

## Technical Specifications

### Data & Processing

- Much of the data processing techniques employed in this article have been taken from the work by Plasencia and his thorough literature review and overview of the procedures for processing medical images that can be seen [here](https://towardsdatascience.com/alzheimer-diagnosis-with-deep-learning-data-preprocessing-4521d6e6ebeb). 
- This section will briefly cover the tools employed and processing techniques and the above link serves as a much more thorough explanation.  
- The data used in this article came from the Alzheimer's Disease Neuroimaging Initiative (ADNI) [database](https://adni.loni.usc.edu/study-design/). ADNI is a global research study in Alzheimer's and the biomarkers that inform us about the disease. Besides providing the data, ADNI has not participated in the project. 
- Two methods used in image preprocessing are image registration and skull stripping. Image registration involved adapting an image to a reference image, often called an atlas, which makes it simpler for a Convolutional Neural Network (CNN) to process. On the other hand, skull stripping is the process of removing information from the skull that appears in MRI images, in order to obtain a clean image as Alzheimer's disease (AD) biomarkers are not found in the skull. A detailed explanation of these steps follows this section. 
- **Python tools employed**: 
    - SimpleITK: part of the SimpleElastix module, used to transform the data from ADNI which were in .nii images into numpy arrays and process them.
    - FSL: used for skull stripping.
    - Libraries like numpy, pandas, matplotlib, and FSL interface in Nipype. 
- **Spatial Normalization**:
    - The spatial normalization preprocessing step is used to ensure that all images in a dataset have a consistent spatial structure.
    - This involves resampling the images to a common isotropic resolution and registering them to a reference atlas. 
    - The resampling and registration methods are defined based on the desired resolution and transformation type.
    - The preprocessing flow typically involves loading the original images, extracting their labels, resampling and registering them to the atlas, and saving the processed images to appropriate destination folders based on their labels. This can be implemented using loops to iterate through all images in the dataset. The specific implementation may vary depending on the libraries and methods used in the code.
- **Skull Stripping**:
    - Skull stripping was the second step in image preprocessing after all the images were spatially normalized. The goal of Skull stripping was to remove irrelevant information from the images and leave only the brain tissue. FSL BET (FMRIB Software Library) interface running on Nipype was used for that specific task. 

## The Way Forward