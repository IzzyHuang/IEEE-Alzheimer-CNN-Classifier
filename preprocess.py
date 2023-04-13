import os
#import csv
import pandas as pd
import image_preprocess

def main():
    
    CSV = "/mnt/d/ADNI/description.csv"
    DATABASE = "/mnt/d/ADNI/RAW/"
    DB_SUBFOLDERS = ['1/', '2/', '3/', '4/', '5/', '6/',
                    '7/', '8/', '9/', '10/']

    REG_DB = "/mnt/d/ADNI/REG"
    SKULL_STRIPPED_DB = "/mnt/d/ADNI/SS"
    CLASS_FOLDERS = ['CN/', 'MCI/', 'AD/']
    ATLAS = "/mnt/d/ADNI/mn305_atlas.nii"

    df = pd.read_csv(CSV)

    process_class = image_preprocess.ImagePreprocess(ATLAS)

    # register all images
    for subdir in DB_SUBFOLDERS:
        for (path, dirs, files) in os.walk(DATABASE + subdir):
            if files:
                for file in files:
                    id = file.split("_")[-1][:-4]
                    row = df.loc[df['Image Data ID'] == id] 
                    group = row.iloc[0]['Group']
                    try:
                        process_class.register_and_save(file, path, group)
                    except RuntimeError:
                        print('Exception with', os.path.join(path, file))

    # applying skull stripping to all registered images
    exceptions = []
    for folder in CLASS_FOLDERS:
        origin_folder = os.path.join(REG_DB, folder)
        dest_folder = os.path.join(SKULL_STRIPPED_DB, folder)
        for path, _, files in os.walk(origin_folder):
            for file in files:
                try:
                    img = os.path.join(path, file)
                    dest = os.path.join(dest_folder, file)
                    process_class.skull_strip_nii(img, dest, frac=0.2)
                except RuntimeError:
                    exceptions.append(img)

    # save the exceptions in case you want to do something about them
    # in our case, FSL BET failed with a couple of images, although it
    # was a very small amount so they were simply discarded
    with open(os.path.join(SKULL_STRIPPED_DB, 'exceptions.txt'), 'w') as f:
        for item in exceptions:
            f.write("%s\n" % item)
            
if __name__ == "__main__":
    main()