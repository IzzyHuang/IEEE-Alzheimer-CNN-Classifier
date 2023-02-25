import os
import csv
import pandas as pd

def count_raw_data(path="D:\ADNI\RAW"):
    total = 0
    for root, dirs, files in os.walk(path):
        total += len(files)
    return total

def get_category_info(dir="D:\ADNI\RAW",csv_path = "D:\ADNI\description.csv"):
    category = {}
    categories = ["MCI","AD","CN"]
    for label in categories:
        category[label] = 0
    df = pd.read_csv(csv_path)
    for root, dirs, files in os.walk(dir):
        for file in files:
            if "ADNI" not in file:
                continue
            id = file.split("_")[-1][:-4]
            #row = df["Image Data ID"]

if __name__ == "__main__":
    path = "D:\ADNI\RAW"
    csv_path = "D:\ADNI\description.csv"
    print(count_raw_data(path))
    get_category_info(path,csv_path)

