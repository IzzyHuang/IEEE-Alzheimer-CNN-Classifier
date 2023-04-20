import streamlit as st
import random
import string
import os
import time


def get_random_string(length):
    result_str = ''.join(random.choice(string.ascii_letters)
                         for i in range(length))
    return result_str


def store_nifti_data(file, temp_data_directory):
    alert = st.warning('Loading data from NIfTI file.')

    # Save NIfTI file to temporary directory
    file_path = os.path.join(temp_data_directory, file.name)
    with open(file_path, 'wb') as out:
        out.write(file.getbuffer())

    # Check if file is a NIfTI file
    if not (file.name.endswith('.nii.gz') or file.name.endswith('.nii')):
        st.warning('Not a valid NIfTI file.', icon="⚠️")
        os.remove(file_path)
        return False
    
    time.sleep(1.5)
    
    alert.empty()

    return True


def does_file_have_nifti_extension(file):
    if not (file.name.endswith('.nii.gz') or file.name.endswith('.nii')):
        st.warning('Not a valid NIfTI file.', icon="⚠️")
        return False
    return True
