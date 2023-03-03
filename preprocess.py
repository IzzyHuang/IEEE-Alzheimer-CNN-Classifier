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

df = pd.read_csv(CSV)

# register all images
for subdir in DB_SUBFOLDERS:
    for path, dirs, files in os.walk(DATABASE + subdir):
        if files:
            for file in files:
                id = file.split("_")[-1][:-4]
                row = df.loc[df['Image Data ID'] == id] 
                group = row.iloc[0]['Group']
                try:
                    register_and_save(file, path, label)
                except RuntimeError:
                    print('Exception with', os.path.join(path, file))