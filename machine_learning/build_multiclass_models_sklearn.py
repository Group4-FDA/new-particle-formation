from utils import \
    scale_features, print_stdout_and_file, plot_confusion_matrix, perplexity
from build_models_sklearn_template import \
    BuildModelsSklearnTemplate
from sklearn.metrics import \
    accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.decomposition import PCA
np.random.seed(42)


class BuildMulticlassModelsSklearn(BuildModelsSklearnTemplate):
    def __init__(
        self,
        input_train_csv_file_name: str,
        input_test_csv_file_name: str,
        target_column: str,
        output_file_name: str,
        columns_to_drop: list[str] = [],
        test_factor: float = 0.2,
        train_folds: int = 5,
        tuning_iterations: int = 20,
        pca=False,
    ):
        BuildModelsSklearnTemplate.__init__(
            self,
            input_train_csv_file_name,
            input_test_csv_file_name,
            target_column,
            output_file_name,
            test_factor=test_factor,
            train_folds=train_folds,
            tuning_iterations=tuning_iterations,
        )
        self.columns_to_drop = columns_to_drop
        self.pca = pca

    def _do_at_init(self) -> None:
        self.df_train.drop(columns=self.columns_to_drop, inplace=True)
        self.df_test.drop(columns=self.columns_to_drop, inplace=True)

    def _do_preprocessing(self) -> None:
        self.x_train, self.x_test \
            = scale_features(self.x_train, self.x_test, self.x_test.columns)
        self.x_train, self.x_test \
            = self.x_train.values, self.x_test.values
        if self.pca:
            pca = PCA(n_components=19, random_state=1)
            self.x_train = pca.fit_transform(self.x_train)
            self.x_test = pca.transform(self.x_test)

    def _do_print_evaluations(
        self,
        train_predictions: list,
        test_predictions: list,
        model_name: str,
        train_predictions_proba: list,
        test_predictions_proba: list,
    ) -> None:
        print_stdout_and_file(
            '\t\t\taccuracy on train: '
            f'{accuracy_score(self.y_train.values, train_predictions)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmacro f1 on train: '
            f'{f1_score(self.y_train.values, train_predictions, average="macro")}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmicro f1 on train: '
            f'{f1_score(self.y_train.values, train_predictions, average="micro")}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\taccuracy on test: '
            f'{accuracy_score(self.y_test.values, test_predictions)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmacro precision on test: '
            f'{precision_score(self.y_test.values, test_predictions, average="macro", zero_division=0)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmicro precision on test: '
            f'{precision_score(self.y_test.values, test_predictions, average="micro", zero_division=0)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmacro recall on test: '
            f'{recall_score(self.y_test.values, test_predictions, average="micro", zero_division=0)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmicro recall on test: '
            f'{recall_score(self.y_test.values, test_predictions, average="micro", zero_division=0)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmacro f1 on test: '
            f'{f1_score(self.y_test.values, test_predictions, average="macro", zero_division=0)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tmicro f1 on test: '
            f'{f1_score(self.y_test.values, test_predictions, average="micro", zero_division=0)}',
            self.file_pointer
        )
        plot_confusion_matrix(
            self.y_test.values,
            test_predictions,
            self.df_train[self.target_column].unique(),
            f"multiclass_{model_name}_average_confusion_matrix.png",
        )
        print_stdout_and_file("", self.file_pointer)
        binary_y_train_actual = [
            'nonevent' if x == 'nonevent' else 'event'
            for x in self.y_train.values
        ]
        binary_y_train_predicted = [
            'nonevent' if x == 'nonevent' else 'event'
            for x in train_predictions
        ]
        binary_y_test_actual = [
            'nonevent' if x == 'nonevent' else 'event'
            for x in self.y_test.values
        ]
        binary_y_test_predicted = [
            'nonevent' if x == 'nonevent' else 'event'
            for x in test_predictions
        ]
        positive_label = 'event'
        print_stdout_and_file(
            '\t\t\tbinary accuracy on train: '
            f'{accuracy_score(binary_y_train_actual, binary_y_train_predicted)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary f1 on train: '
            f'{f1_score(binary_y_train_actual, binary_y_train_predicted, pos_label=positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary perplexity train: '
            f'{perplexity(train_predictions_proba, binary_y_train_predicted, positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary accuracy on test: '
            f'{accuracy_score(binary_y_test_actual, binary_y_test_predicted)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary precision on test: '
            f'{precision_score(binary_y_test_actual, binary_y_test_predicted, pos_label=positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary precision on test: '
            f'{precision_score(binary_y_test_actual, binary_y_test_predicted, pos_label=positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary recall on test: '
            f'{recall_score(binary_y_test_actual, binary_y_test_predicted, pos_label=positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary f1 on test: '
            f'{f1_score(binary_y_test_actual, binary_y_test_predicted, pos_label=positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tbinary perplexity test: '
            f'{perplexity(test_predictions_proba, binary_y_test_predicted, positive_label)}',
            self.file_pointer
        )
        plot_confusion_matrix(
            self.y_test.values,
            test_predictions,
            self.df_train[self.target_column].unique(),
            f"multiclass_binary_{model_name}_average_confusion_matrix.png",
        )


normal_process = BuildMulticlassModelsSklearn(
    input_train_csv_file_name='resources/data/generated/train_train.csv',
    input_test_csv_file_name='resources/data/generated/train_test.csv',
    target_column='class4',
    output_file_name=(
        'resources/machine_learning_results/'
        'multiclass_classification_models.txt'
    ),
    columns_to_drop=['id', 'date', 'partlybad'],
    test_factor=0.2,
    train_folds=5,
    tuning_iterations=20,
)
process_pca = BuildMulticlassModelsSklearn(
    input_train_csv_file_name='resources/data/generated/train_train.csv',
    input_test_csv_file_name='resources/data/generated/train_test.csv',
    target_column='class4',
    output_file_name=(
        'resources/machine_learning_results/'
        'multiclass_classification_models_pca.txt'
    ),
    columns_to_drop=['id', 'date', 'partlybad'],
    test_factor=0.2,
    train_folds=5,
    tuning_iterations=20,
    pca=True
)
process_pca.compute()
normal_process.compute()
