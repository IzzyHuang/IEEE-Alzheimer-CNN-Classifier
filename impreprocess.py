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
    def __init__(self, old_path = None, dest_path = None):

        # path to image 
        self.old_path = old_path

        # where the new image will be stored 
        self.dest_path = dest_path

        # this line loads the images in sitk format - read image 
        self.sitk_im = sitk.ReadImage(old_path)

        # transform into a numpy array 
        self.img = sitk.GetArrayFromImage(self.sitk_image)

    def resample_img(self, out_spacing=[2.0, 2.0, 2.0]):
        ''' This function resamples images to 2-mm isotropic voxels.
        
            Parameters:
                itk_image -- Image in simpleitk format, not a numpy array
                out_spacing -- Space representation of each voxel
                
            Returns: 
                Resulting image in simpleitk format, not a numpy array
        '''
        
        # Resample images to 2mm spacing with SimpleITK
        original_spacing = self.sitk_im.GetSpacing()
        original_size = self.sitk_im.GetSize()

        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(self.sitk_im.GetDirection())
        resample.SetOutputOrigin(self.sitk_im.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(self.sitk_im.GetPixelIDValue())

        resample.SetInterpolator(sitk.sitkBSpline)

        return resample.Execute(self.sitk_im)

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


    def run(self):
        pass


