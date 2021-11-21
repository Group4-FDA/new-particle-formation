"""
Use the buildings_model_features csv to build models using sklearn

Here we do cross validation and hyperparameter tuning for each
model that we train, using sklearn
"""

import warnings

from models import classification_models as models
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import \
    accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from utils import (
    split_in_folds_classification,
    extract_target_feature,
    scale_features,
    print_stdout_and_file,
    plot_confusion_matrix,
)
np.random.seed(42)
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning


to_drop = ['id', 'date', 'partlybad']

file_pointer \
    = open('resources/machine_learning_results/classification_models.txt', 'w')

fold_amount = 5
target_value = 'class4'

df = pd.read_csv('resources/data/original/npf_train.csv')
df.drop(columns=to_drop, inplace=True)

positive_label = 'event'
df['class4'] = df['class4'].map(
    lambda x: positive_label if x != 'nonevent' else x
)

# split into x_train, y_train, x_test, y_test
folds = split_in_folds_classification(df, fold_amount, target_value)
x_test = folds[0]
x_train = pd.concat(folds[1:])
x_train, y_train = extract_target_feature(x_train, target_value)
x_test, y_test = extract_target_feature(x_test, target_value)

# scale all features
x_train, x_test = scale_features(x_train, x_test, x_test.columns)

for name, model in models.items():
    print_stdout_and_file(f'Now training {name}', file_pointer)
    # Hyper parameter tuning on the train set
    print('\tTuning...')
    tuner = RandomizedSearchCV(
        model['class'](**model['set_parameters']),
        model['hyperparameters'],
        refit=False,
        n_iter=20,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        tuner.fit(x_train.values, y_train.values)
    best_params = tuner.best_params_ | model['set_parameters']
    print_stdout_and_file(f'\t\tBest Params: {best_params}', file_pointer)

    # train and evaluate
    classifier = model['class'](**best_params)
    print_stdout_and_file('\t\tFitting...', file_pointer)
    classifier.fit(x_train.values, y_train.values)
    train_predictions = classifier.predict(x_train.values)
    test_predictions = classifier.predict(x_test.values)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        print_stdout_and_file(
            '\t\t\taccuracy on train: '
            f'{accuracy_score(y_train.values, train_predictions)}',
            file_pointer
        )
        print_stdout_and_file(
            '\t\t\tf1 on train: '
            f'{f1_score(y_train.values, train_predictions, pos_label=positive_label)}',
            file_pointer
        )
        print_stdout_and_file(
            '\t\t\taccuracy on test: '
            f'{accuracy_score(y_test.values, test_predictions)}',
            file_pointer
        )
        print_stdout_and_file(
            '\t\t\tprecision on test: '
            f'{precision_score(y_test.values, test_predictions, pos_label=positive_label)}',
            file_pointer
        )
        print_stdout_and_file(
            '\t\t\tprecision on test: '
            f'{precision_score(y_test.values, test_predictions, pos_label=positive_label)}',
            file_pointer
        )
        print_stdout_and_file(
            '\t\t\trecall on test: '
            f'{recall_score(y_test.values, test_predictions, pos_label=positive_label)}',
            file_pointer
        )
        print_stdout_and_file(
            '\t\t\tf1 on test: '
            f'{f1_score(y_test.values, test_predictions, pos_label=positive_label)}',
            file_pointer
        )
        print_stdout_and_file("", file_pointer)
        plot_confusion_matrix(
            y_test.values,
            test_predictions,
            df[target_value].unique(),
            f"binary_{name}_average_confusion_matrix.png",
        )

file_pointer.close()
