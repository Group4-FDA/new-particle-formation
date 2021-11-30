import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'/Users/sini/Documents/Intro to ML/Project/new-particle-formation-master/resources/data/generated/train_train.csv')

# get only columns with mean
feature = df.drop(columns=['id', 'date', 'class4', 'partlybad'])

# Scale data before applying PCA
scaler = StandardScaler()
data_rescaled = scaler.fit_transform(feature)

pca = PCA().fit(data_rescaled)

#plot a graph
plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, 102, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 102, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()

#choose number of component as 19
pca = PCA(n_components=19 , random_state=1)
pca.fit(data_rescaled)
x_pca = pca.transform(data_rescaled)

#explained variance tells how much infomation (variance) can be attributed to each of the principal components
pca.explained_variance_ratio_

print([(i+1,a) for i, a in enumerate(y)])