"""
separate pictures into folders named by breed
"""

import os
import pandas as pd

dataset_path = "data/kaggle_dataset/"
train_path = dataset_path + "train/"
test_path = dataset_path + "test/"
labels_path = dataset_path + "labels.csv"

labels: pd.Series = pd.read_csv(labels_path, index_col = 0, squeeze = True)


def separate_by_breed(root_dir, labels):
    cwd = os.getcwd()
    os.chdir(root_dir)

    filenames = set([filename[:-4] for filename in os.listdir()]) # all filenames without .jpg extensions

    for (filename, breed) in labels.iteritems():
        if filename in filenames:
            if os.path.isdir(breed):
                print(breed)
            
            else:
                os.mkdir(breed)
            
            os.rename(filename + ".jpg", f"{breed}/{filename}.jpg")
    
    os.chdir(cwd)


separate_by_breed(train_path, labels)