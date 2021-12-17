"""
This script divides the original dataset into its own train and test set
The split is 80% train set and 20% for the test set
"""

import sys
sys.path += ['machine_learning']

import numpy as np
import pandas as pd
from utils import get_stratified_train_test_folds
np.random.seed(42)


df = pd.read_csv('resources/data/original/npf_train.csv')
train_df, test_df = \
    get_stratified_train_test_folds(df, 'class4', test_factor=0.2)
train_df.to_csv('resources/data/generated/train_train.csv', index=False)
test_df.to_csv('resources/data/generated/train_test.csv', index=False)
