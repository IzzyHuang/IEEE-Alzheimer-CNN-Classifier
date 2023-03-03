import os
import csv
import pandas as pd

CSV = "D:\ADNI\description.csv"
DATABASE = "D:\ADNI\RAW"
DB_SUBFOLDERS = ['1/', '2/', '3/', '4/', '5/', '6/',
                 '7/', '8/', '9/', '10/']

REG_DB = "D:\ADNI\REG"
SKULL_STRIPPED_DB = "D:\ADNI\SS"
CLASS_FOLDERS = ['CN/', 'MCI/', 'AD/']
ATLAS = "D:\ADNI\mn305_atlas.nii"

# prepare and organize the images
for subdir in DB_SUBFOLDERS:
    for path, dirs, files in os.walk(DATABASE + subdir):
        if files:
            for file in files:
                try:
                    register_and_save(file, path, ATLAS)
                except RuntimeError:
                    print('Exception with', os.path.join(path, file))