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
)
np.random.seed(42)


class BuildModelsSklearnTemplate:
    def __init__(
        self,
        input_train_csv_file_name: str,
        input_test_csv_file_name: str,
        target_column: str,
        output_file_name: str,
        test_factor: float = 0.2,
        train_folds: int = 5,
        tuning_iterations: int = 20,
    ):
        self.file_pointer = open(output_file_name, 'w')
        self.df_train = pd.read_csv(input_train_csv_file_name)
        self.df_test = pd.read_csv(input_test_csv_file_name)
        self.target_column = target_column
        self.test_factor = test_factor
        self.train_folds = train_folds
        self.tuning_iterations = tuning_iterations

    def _do_at_init(self) -> None:
        pass

    def _initialize_train_test_split(self) -> None:
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
            tuner.fit(self.x_train, self.y_train)
        return tuner.best_params_ | set_parameters

    def _train_model(
        self, model: sklearn.base.BaseEstimator, parameters: dict
    ) -> sklearn.base.BaseEstimator:
        classifier = model['class'](**parameters)
        classifier.fit(self.x_train, self.y_train)
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
            print('\t\tFitting...')
            classifier = self._train_model(model, best_params)
            train_predictions = classifier.predict(self.x_train)
            test_predictions = classifier.predict(self.x_test)
            non_event_index = list(classifier.classes_).index('nonevent')
            train_predictions_proba = \
                1 - classifier.predict_proba(self.x_train)[:, non_event_index]
            test_predictions_proba = \
                1 - classifier.predict_proba(self.x_test)[:, non_event_index]
            self._do_print_evaluations(
                train_predictions,
                test_predictions,
                name,
                train_predictions_proba,
                test_predictions_proba,
            )
        self.file_pointer.close()
