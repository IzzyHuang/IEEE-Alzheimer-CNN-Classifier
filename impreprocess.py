import SimpleITK as sitk #not importing    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from dltk.io import preprocessing #not importing
from skimage import filters

import os

# from nipype.interfaces import fsl

# # check the final shape
# img.shape

# plt.imshow(img[:, :, 70], cmap='gray')
# plt.show()

# otsu = filters.threshold_otsu(img)
# otsu_img = img > otsu
# plt.imshow(otsu_img[:, :, 70], cmap='gray')
# plt.show()

# res = resample_img(sitk_image)
# res_img = sitk.GetArrayFromImage(sitk_image)
# res_img = preprocessing.resize_image_with_crop_or_pad(res_img, img_size=(128, 192, 192), mode='symmetric')
# res_img = preprocessing.whitening(res_img)
# plt.imshow(res_img[:, 100, :], cmap='gray')
# plt.show()

# registered and organized database
REG_DB = '/content/Regeistered'
# the images should be divided by its label
REG_DB_SUBFOLDERS = ['AD/', 'MCI/', 'CN/']


class ImagePreprocess:
    def __init__(self, path_to_atlas = None):

        # path to atlas 
        self.path_to_atlas = path_to_atlas

        # atlas 
        self.atlas = sitk.ReadImage(self.path_to_atlas)



    def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0]):
        ''' This function resamples images to 2-mm isotropic voxels (default).
        
            Parameters:
                itk_image -- Image in simpleitk format, not a numpy array
                out_spacing -- Space representation of each voxel
                
            Returns: 
                Resulting image in simpleitk format, not a numpy array
        '''
        
        # Resample images to 2mm spacing with SimpleITK
        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()

        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

        resample.SetInterpolator(sitk.sitkBSpline)

        return resample.Execute(itk_image)

    def registrate(self, sitk_fixed, sitk_moving, bspline=False):
        ''' Perform image registration using SimpleElastix.
            By default, uses affine transformation.
            
            Parameters:
                sitk_fixed -- Reference atlas (sitk .nii)
                sitk_moving -- Image to be registrated
                            (sitk .nii)
                bspline -- Whether or not to perform non-rigid
                        registration. Note: it usually deforms
                        the images and increases execution times
        '''
        
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(sitk_fixed)
        elastixImageFilter.SetMovingImage(sitk_moving)

        parameterMapVector = sitk.VectorOfParameterMap()
        parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
        if bspline:
            parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
        elastixImageFilter.SetParameterMap(parameterMapVector)

        elastixImageFilter.Execute()
        return elastixImageFilter.GetResultImage()

    def register_and_save(self, filename, path, atlas, label):
        ''' Process the image name and copy the image to its
            corresponding destination folder.
            
            Parameters:
                filename -- Name of the image file (.nii)
                path -- The path were the image is located
                atlas -- Reference sitk image for registration
        '''
        
        # separate the name of the file by '_'
        splitted_name = filename.strip().split('_')
        # sometimes residual MacOS files appear; ignore them
        if splitted_name[0] == '.': return
        
        # save the image ID
        image_ID = splitted_name[-1][1:-4]
        
        # sometimes empty files appear, just ignore them (macOS issue)
        if image_ID == '': return
        # transform the ID into a int64 numpy variable for indexing
        image_ID = np.int64(image_ID)
            
        #### IMPORTANT #############
        # the following three lines are used to extract the label of the image
        # ADNI data provides a description .csv file that can be indexed using the
        # image ID. If you are not working with ADNI data, then you must be able to 
        # obtain the image label (AD/MCI/NC) in some other way
        # with the ID, index the information we need

        # row_index = description.index[description['Image Data ID'] == image_ID].tolist()[0]

        # # obtain the corresponding row in the dataframe

        # row = description.iloc[row_index]

        # # get the label

        # label = row['Group']
        
        # prepare the origin path
        complete_file_path = os.path.join(path, filename)
        # load sitk image
        sitk_moving = sitk.ReadImage(complete_file_path)
        sitk_moving = self.resample_img(sitk_moving)
        registrated = self.registrate(atlas, sitk_moving)
        
        # prepare the destination path
        complete_new_path = os.path.join(REG_DB, 
                                        label,
                                        filename)
        sitk.WriteImage(registrated, complete_new_path)

    def run(self):
        pass


