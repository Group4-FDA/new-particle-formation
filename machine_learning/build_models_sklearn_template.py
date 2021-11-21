"""
Use the buildings_model_features csv to build models using sklearn

Here we do cross validation and hyperparameter tuning for each
model that we train, using sklearn

This class uses the Template Design Pattern
"""

import warnings

from models import classification_models as models
import sklearn
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from utils import (
    split_in_folds_classification,
    extract_target_feature,
    print_stdout_and_file,
    get_stratified_train_test_folds,
)
np.random.seed(42)
from sklearn.exceptions import ConvergenceWarning


class BuildModelsSklearnTemplate:
    def __init__(
        self,
        input_csv_file_name: str,
        target_column: str,
        output_file_name: str,
        test_factor: float = 0.2,
        train_folds: int = 5,
        tuning_iterations: int = 20,
    ):
        self.file_pointer = open(output_file_name, 'w')
        self.df = pd.read_csv(input_csv_file_name)
        self.target_column = target_column
        self.test_factor = test_factor
        self.train_folds = train_folds
        self.tuning_iterations = tuning_iterations

    def _do_at_init(self) -> None:
        pass

    def _initialize_train_test_split(self) -> None:
        self.df_train, self.df_test = get_stratified_train_test_folds(
            self.df, self.target_column, self.test_factor
        )
        self.x_train, self.y_train \
            = extract_target_feature(self.df_train, self.target_column)
        self.x_test, self.y_test \
            = extract_target_feature(self.df_test, self.target_column)

    def _do_preprocessing(self) -> None:
        pass

    def _initialize_folds(self) -> None:
        self.folds = split_in_folds_classification(
            self.df_train, self.train_folds, self.target_column
        )

    def _tune_model(
        self,
        model: sklearn.base.BaseEstimator,
        set_parameters: dict,
        hyper_parameters: dict,
    ) -> dict:
        tuner = RandomizedSearchCV(
            model,
            hyper_parameters,
            refit=False,
            n_iter=20,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            tuner.fit(self.x_train.values, self.y_train.values)
        return tuner.best_params_ | set_parameters

    def _train_model(
        self, model: sklearn.base.BaseEstimator, parameters: dict
    ) -> sklearn.base.BaseEstimator:
        classifier = model['class'](**parameters)
        classifier.fit(self.x_train.values, self.y_train.values)
        return classifier

    def _do_print_evaluations(
        self, train_predictions: list, test_predictions: list
    ) -> None:
        pass

    def compute(self) -> None:
        self._do_at_init()
        self._initialize_train_test_split()
        self._do_preprocessing()
        self._initialize_folds()
        for name, model in models.items():
            print_stdout_and_file(f'Now training {name}', self.file_pointer)
            print('\tTuning...')
            best_params = self._tune_model(
                model['class'](**model['set_parameters']),
                model['set_parameters'],
                model['hyperparameters'],
            )
            print_stdout_and_file(
                f'\t\tBest Params: {best_params}', self.file_pointer
            )
            print_stdout_and_file('\t\tFitting...', self.file_pointer)
            classifier = self._train_model(model, best_params)
            train_predictions = classifier.predict(self.x_train.values)
            test_predictions = classifier.predict(self.x_test.values)
            self._do_print_evaluations(
                train_predictions, test_predictions, name
            )
        self.file_pointer.close()
