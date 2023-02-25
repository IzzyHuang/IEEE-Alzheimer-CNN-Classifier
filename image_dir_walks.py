import os
import csv
import pandas as pd

def count_raw_data(path="D:\ADNI\RAW"):
    total = 0
    for root, dirs, files in os.walk(path):
        total += len(files)
    return total


if __name__ == "__main__":
    path = "D:\ADNI\RAW"
    csv_path = "D:\ADNI\description.csv"
    print(count_raw_data(path))
    #get_category_info(path,csv_path)

