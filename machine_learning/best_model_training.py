"""
This script simply repeats the steps in the template file,
but fits the model on the entire train set once and applies
it to the test set to get the predictions for that
"""

import pandas as pd
from utils import extract_target_feature, scale_features
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
np.random.seed(42)


target_column = 'class4'

params = {
    'C': 1, 'max_iter': 9999, 'class_weight': 'balanced', 'solver': 'liblinear'
}
classifier = LogisticRegression(**params)

df_train = pd.read_csv('resources/data/original/npf_train.csv')
df_test = pd.read_csv('resources/data/original/npf_test_hidden.csv')

# init
for d in [df_train, df_test]:
    d.drop(columns='id date partlybad'.split(), inplace=True)

x_train, y_train \
    = extract_target_feature(df_train, target_column)
x_test, y_test \
    = extract_target_feature(df_test, target_column)

# preprocessing
x_train, x_test = scale_features(x_train, x_test, x_test.columns)
x_train, x_test = x_train.values, x_test.values

pca = PCA(n_components=19, random_state=1)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)
non_event_index = list(classifier.classes_).index('nonevent')
test_predictions_proba = \
    1 - classifier.predict_proba(x_test)[:, non_event_index]

df = pd.DataFrame.from_dict(
    {'class4': predictions, 'p': test_predictions_proba}
)
df.to_csv('resources/machine_learning_results/answers.csv', index=False)
