# NeuroNet - A Deep Learning Model for Alzheimer's Diagnosis 

## Introduction

This project is a deep learning model that can diagnose Alzheimer's Disease through MRI data. Moreover, it can even detect Alzheimer's early, in the form of early mild cognitive impairment and late mild cognitive impairment. While this model isn't meant to be a definitive test to diagnose Alzheimer's, it can serve as an initial diagnostic tool. 

The model is a convolutional neural network that analyzes 3D MRI images. A much more thorough explanation of the technical specifications, including image preprocessing and the model training is in the next few sections.

This project was inspired by the wonderful work conducted by Oscar Darias Plasencia that can be seen [here](https://towardsdatascience.com/alzheimer-diagnosis-with-deep-learning-a-survey-265406fa542a). 

P.S - the combination of medical imaging libraries and ML packages meant that the project in its entirety could not be run on any on system. Therefore, the files are littered with system specific constants and methods. Should you choose to play with the model, you could download the file from [here](https://drive.google.com/file/d/1KPfZXW8-9cdQqYKw4PNOimw4P23CXg1Y/view?usp=sharing) and look at the client file for how to interface with it. The requirement.txt could serve as a starting point for the libraries you would need.

## Technical Specifications

### Data & Processing

- Much of the data processing techniques employed in this article have been taken from the work by Plasencia and his thorough literature review and overview of the procedures for processing medical images that can be seen [here](https://towardsdatascience.com/alzheimer-diagnosis-with-deep-learning-data-preprocessing-4521d6e6ebeb). 
- This section will briefly cover the tools employed and processing techniques and the above link serves as a much more thorough explanation.  
- The data used in this article came from the Alzheimer's Disease Neuroimaging Initiative (ADNI) [database](https://adni.loni.usc.edu/study-design/). ADNI is a global research study in Alzheimer's and the biomarkers that inform us about the disease. Besides providing the data, ADNI has not participated in the project. 
- The ADNI dataset is broken up into 3 labelled categories, "CN", "MCI" & "AD". These stand for cognitively normal, mild cognitive impairment (early or late, both of which are indicators of Alzheimer's) and Alzheimer's Disease.
- The dataset had the following breakup:
{'MCI': 1112, 'AD': 476, 'CN': 705} with 2293 total 3D MRI images. 
- We randomly segmented this dataset into training, validation and testing data with a 0.7, 0.15 and 0.15 ratio respectively.
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

- Finally, the processed data was converted into a TFRecords type to account for limitations on storage with Google Colab.

### Model & Training

- For the project, we chose to use a Convolutional Neural Network (CNN) to analyze the MRI information from the ADNI study. We set up our model to give us a binary output, the probability that an MRI falls into the CN category or the AD category. Since the MCI category represents early AD, we chose to treat those cases as such, in order to account for MRIs that present early symptoms of Alzheimer's Disease.
- A Convolutional Neural Network (CNN) is a deep learning algorithm used for image and video analysis. It is modeled after the way the human brain processes visual information. The network is made up of several layers that are responsible for identifying patterns in the input image. The convolutional layers use filters to extract useful features from the image, the pooling layers downsample the feature maps to reduce dimensionality, and the fully connected layers classify the image based on the features extracted in the previous layers. CNNs have achieved remarkable success in many computer vision tasks, such as object detection, recognition, and segmentation.
- **Model Structure**:
    - Our CNN model starts by initializing with the Sequential function. The first layer added is a Conv2D layer with 32 filters, a 3x3 kernel size, and Rectified Linear Unit (ReLU) activation function. Next, a MaxPooling2D layer with a 2x2 pool size is added to reduce the spatial dimensions of the output from the previous layer. This is repeated with another Conv2D layer with 64 filters and another MaxPooling2D layer. The Flatten function is used to convert the output of the convolutional layers into a one-dimensional vector. A dense layer with 64 neurons and ReLU activation is added, followed by an output layer with two neurons using the sigmoid activation function. The model is compiled with the Adam optimizer, binary cross-entropy loss, and accuracy metrics. Finally, the model is fit to the data using the training and validation datasets with 20 epochs, batch size, and steps per epoch specified.

    ![Model Structure](https://github.com/srsavas42/IE3_ML/blob/main/resources/model.png)

- **Quality Analysis**:
    - With any ML model, it is quite difficult to definitively ascertain its accuracy. Owing to the imbalances in our classes, we chose to use the Receiver Operating Characteristic - Area Under the Curve (ROC AUC). 
    - This performance metric is often used to evaluate the ability of a binary classification model to distinguish between positive and negative classes. 
    - The ROC curve is a graph that shows the trade-off between senstivity (true positive rate) and specificity (true negative rate) at different thresholds. The area under the ROC curve represents the model's ability to correctly classify positive and negative examples across all possible thresholds.
    - In simple terms, a higher ROC AUC score indicates that the model has better discrimination power, meaning it can more accurately distinguish between positive and negative examples. A perfect ROC AUC score is 1, which means that the model can perfectly separate positive and negative examples. Conversely, a score of 0.5 indicates that the model performs no better than random guessing.
    - Our model displayed an ROC AUC of 0.94, which means that our model has an accuracy of roughly 94% in correctly classifying positive and negative examples. 

    ![ROC AUC](https://github.com/srsavas42/IE3_ML/blob/main/resources/roc_auc.png)

### Client

## The Way Forward