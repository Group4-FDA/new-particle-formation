import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_pca19 = pd.read_csv(r'/Users/sini/Documents/Intro to ML/Project/new-particle-formation-master/resources/data/generated/pca19_train_train.csv')
train_train = pd.read_csv(r'/Users/sini/Documents/Intro to ML/Project/new-particle-formation-master/resources/data/generated/train_train.csv')

train_train = train_train.drop(columns=['id', 'date', 'class4', 'partlybad'])
train_pca19 = train_pca19.drop(columns=['Unnamed: 0'])

concat = pd.concat([train_pca19, train_train], axis=1)

f = plt.figure(figsize=(25, 25))
plt.matshow(concat.corr(), fignum=f.number)
plt.xticks(range(concat.select_dtypes(['number']).shape[1]), concat.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(concat.select_dtypes(['number']).shape[1]), concat.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)