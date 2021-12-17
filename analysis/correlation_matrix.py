"""
This script generates a correlation matrix between the pca features
and the original features
"""

import pandas as pd
import matplotlib.pyplot as plt

train_pca19 = pd.read_csv('resources/data/generated/pca19_train_train.csv')
train_train = pd.read_csv('resources/data/generated/train_train.csv')

train_train = train_train.drop(columns=['id', 'date', 'class4', 'partlybad'])
train_pca19 = train_pca19.drop(columns=['Unnamed: 0'])

concat = pd.concat([train_pca19, train_train], axis=1)

f = plt.figure(figsize=(7, 25))
corr = concat.corr().iloc[19:, :19]
plt.matshow(corr, fignum=f.number)
plt.yticks(ticks=range(len(corr.index)), labels=corr.index.values, fontsize=11)
plt.xticks(ticks=range(len(corr.columns)), labels=corr.columns.values, fontsize=11, rotation=45)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=True,
    labelbottom=False
)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
plt.title('PCA Correlation Matrix', fontsize=16)
plt.savefig('a.png')
