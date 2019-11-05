

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile

DIR = 'C:\\ImageExport'
os.chdir(DIR)
# get the classification categories of the dataset
categories = list(filter(os.path.isdir, os.listdir(os.getcwd())))
categories

# from the folders, extract the filename and the size of the images
file_info=pd.DataFrame()
for category in tqdm(categories):
    path, dirs, files = next(os.walk(os.path.join(DIR, category)))
    
    temp_df = pd.DataFrame({'category':np.repeat(category, len(files)),'filenames': files, 
                            'shape': np.repeat("", len(files))}) 
    row = []
    col = []
    for file in tqdm(files):
        img = cv2.imread(os.path.join(DIR,category, file) ,cv2.IMREAD_GRAYSCALE)
        shape = img.shape
        row.append(shape[0])
        col.append(shape[1])
    temp_df['row'] = row
    temp_df['col'] = col
    file_info = file_info.append(temp_df,'sort=False')

# save the dataframe for safe keeping 
file_info.to_csv('file_info.csv')

# reload the dataframe just saved
file_info = pd.read_csv('file_info.csv')
file_info.drop('Unnamed: 0', axis=1,inplace=True)

# get summary statistics and histogram
file_info.describe()
file_info.describe(include=['O'])

fig = plt.figure(figsize = (8,8))
ax = fig.gca()
file_info.loc[:, ['row', 'col']].hist(ax=ax)
plt.show()

cat_row_count = file_info.groupby('row', as_index=False)['filenames'].count().sort_values('filenames', ascending=0)
cat_col_count = file_info.groupby('row', as_index=False)['filenames'].count().sort_values('filenames', ascending=0)

# divide the dataset into train, validation, and test
splits={}
for category in categories:
    temp_df = file_info.loc[file_info['category']==category,:]
    X_train, X_test, _, _ = train_test_split(temp_df, temp_df['category'], test_size=0.10, random_state=1)
    X_test_index = list(X_test.index)
    X_train1, X_validation, _, _ = train_test_split(X_train, X_train['category'], test_size=0.20, random_state=1)
    X_train_index = list(X_train1.index)
    X_validation_index = list(X_validation.index)
    splits[category] = {'X_train_index': X_train_index, 'X_validation_index': X_validation_index,
          'X_test_index':X_test_index}

# create new directory structure using the images clases
datatypes = ['train', 'validation', 'test']
new_DIR = '../dataset'
for key, val in tqdm(splits.items()):
    for datatype in tqdm(datatypes):
        indexes = val['X_'+datatype+'_index']
        newpath = os.path.join(new_DIR, datatype, key)
        if not(os.path.exists(newpath)):
            print('newpath:', newpath)
            os.makedirs(newpath)
        for index in tqdm(indexes):
            filename = file_info.iloc[index, 1]
            oldfile = os.path.join(DIR, key, filename)
            newfile = os.path.join(newpath, filename)
            copyfile(oldfile, newfile)
            
            
   
        
    
    
    
    
    
    