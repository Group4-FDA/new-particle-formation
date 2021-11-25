import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

np.random.seed(333)

df = pd.read_csv(r'/Users/sini/Documents/Intro to ML/project/npf_train.csv') #need to change to train_train set

feature = df.filter(regex='mean')

scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(feature)

pca = PCA().fit(data_rescaled)

plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, 51, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 51, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()

#choose number of component as 6
pca = PCA(n_components=6)
pca.fit(data_rescaled)
x_pca = pca.transform(data_rescaled)

#explained variance tells how much information (variance) can be attributed to each of the principal components
pca.explained_variance_ratio_