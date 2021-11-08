# classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


pow_10_paramter = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

classification_models = {
    "LogisticRegression": {
        "class": LogisticRegression,
        "hyperparameters": {
            "C": pow_10_paramter,
        },
        "set_parameters": {
            "n_jobs": -1,
            "max_iter": 2000
        }
    },
    "BalancedLogisticRegression": {
        "class": LogisticRegression,
        "hyperparameters": {
            "C": pow_10_paramter,
        },
        "set_parameters": {
            "n_jobs": -1,
            "max_iter": 2000,
            "class_weight": "balanced"
        }
    },
    # "SVM": {
    #     "class": SVC,
    #     "hyperparameters": {
    #         "C": pow_10_paramter,
    #         "kernel": ["linear", "poly", "rbf", "sigmoid"],
    #         "degree": list(range(1, 6, 1)),
    #     },
    #     "set_parameters": {}
    # },
    # "BalancedSVM": {
    #     "class": SVC,
    #     "hyperparameters": {
    #         "C": pow_10_paramter,
    #         "kernel": ["linear", "poly", "rbf", "sigmoid"],
    #         "degree": list(range(1, 3, 1)),
    #     },
    #     "set_parameters": {
    #         "class_weight": "balanced"
    #     }
    # },
    # "MLPClassifier": {
    #     "class": MLPClassifier,
    #     "hyperparameters": {
    #         "hidden_layer_sizes": [
    #             (200, 200), (100, 100), (100, 100, 100), (200, 100, 100)
    #         ],
    #         "activation": ["identity", "logistic", "tanh", "relu"],
    #         "solver": ["lbfgs", "sgd", "adam"],
    #         "alpha": pow_10_paramter,
    #     },
    #     "set_parameters": {
    #         "early_stopping": True,
    #         "max_iter": 500,
    #     }
    # }
}
