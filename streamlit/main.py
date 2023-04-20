import streamlit as st
import streamlit.components.v1 as components
from ipywidgets import embed
import vtk
from itkwidgets import view
from streamlit_lottie import st_lottie
import requests
from utils import does_file_have_nifti_extension, get_random_string, store_nifti_data
from glob import glob
import os


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


global temp_data_directory
temp_data_directory = ''


lottie_file = load_lottieurl(
    'https://assets7.lottiefiles.com/packages/lf20_MRJyJMGGfk.json')
data_key = 'has_data'
data_has_changed = False

hide_menu_style = """
<style>
#MainMenu {visibility: hidden; }
footer {visibility: hidden; }
</style>
"""
st.set_page_config(page_title="NeuroNet")
st.markdown(hide_menu_style, unsafe_allow_html=True)
st_lottie(lottie_file, height=200, key='coding')

st.title("**_NeuroNet_: A Deep Learning Model for Alzheimer's Diagnosis**")
st.write(
    "Upload your **:blue[MRI scan]** and find out if you could be at risk for Alzheimer's.")

input_file = st.file_uploader('Choose a file')

# store the file
if input_file:
    if does_file_have_nifti_extension(input_file):
        temp_data_directory = f'./data/{get_random_string(15)}/'
        os.makedirs(temp_data_directory, exist_ok=True)
        store_nifti_data(input_file, temp_data_directory)
        data_has_changed = True

# show the 3d image
if st.button('View 3D Image'):
    path_to_file = glob(f'{temp_data_directory}/*.nii*')
    if path_to_file:
        with st.container():
            reader = vtk.vtkNIFTIImageReader()
            reader.SetFileName(path_to_file[0])
            reader.Update()

            view_width = 900
            view_height = 800

            snippet = embed.embed_snippet(views=view(reader.GetOutput()))
            html = embed.html_template.format(title="", snippet=snippet)
            components.html(html, width=view_width, height=view_height)
