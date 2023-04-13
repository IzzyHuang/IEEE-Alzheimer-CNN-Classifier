import SimpleITK as sitk  # not importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dltk.io import preprocessing  # not importing
from skimage import filters

import os

from nipype.interfaces import fsl

# registered and organized database
REG_DB = "/mnt/d/ADNI/REG"
SKULL_STRIPPED_DB = "/mnt/d/ADNI/SS"
# the images should be divided by its label
REG_DB_SUBFOLDERS = ['AD', 'MCI', 'CN']


class ImagePreprocess:
    def __init__(self, path_to_atlas="/mnt/d/ADNI/mn305_atlas.nii"):

        # path to atlas
        self.path_to_atlas = path_to_atlas

        # atlas
        self.atlas = sitk.ReadImage(self.path_to_atlas)
        self.atlas = self.resample_img(self.atlas)

    def resample_img(self,itk_image, out_spacing=[2.0, 2.0, 2.0]):
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
            int(np.round(original_size[0] *
                (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] *
                (original_spacing[1] / out_spacing[1]))),
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

    def register_and_save(self, filename, path, label):
        ''' Process the image name and copy the image to its
            corresponding destination folder.

            Parameters:
                filename -- Name of the image file (.nii)
                path -- The path were the image is located
                label -- subgroup for the image, one of "MCI", "AD", "CN"
        '''

        # prepare the origin path
        complete_file_path = os.path.join(path, filename)
        # load sitk image
        sitk_moving = sitk.ReadImage(complete_file_path)
        sitk_moving = self.resample_img(sitk_moving)
        registrated = self.registrate(self.atlas, sitk_moving)

        # prepare the destination path
        complete_new_path = os.path.join(REG_DB,
                                         label,
                                         filename)
        print(f"writing image to {complete_new_path}")
        sitk.WriteImage(registrated, complete_new_path)

    def skull_strip_nii(self,original_img, destination_img, frac=0.2):
        ''' Practice skull stripping on the given image, and save
            the result to a new .nii image.
            Uses FSL-BET 
            (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#Main_bet2_options:)

            Parameters:
                original_img -- Original nii image
                destination_img -- The new skull-stripped image
                frac -- Fractional intensity threshold for BET
        '''

        btr = fsl.BET()
        btr.inputs.in_file = original_img
        btr.inputs.frac = frac
        btr.inputs.out_file = destination_img
        btr.cmdline
        res = btr.run()
