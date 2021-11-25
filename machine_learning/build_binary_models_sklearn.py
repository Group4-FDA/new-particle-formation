from build_models_sklearn_template import \
    BuildModelsSklearnTemplate
from utils import scale_features, print_stdout_and_file, plot_confusion_matrix
from sklearn.metrics import \
    accuracy_score, f1_score, precision_score, recall_score


class BuildBinaryModelsSklearn(BuildModelsSklearnTemplate):
    def __init__(
        self,
        input_csv_file_name: str,
        target_column: str,
        positive_label: str,
        negative_label: str,
        output_file_name: str,
        columns_to_drop: list[str] = [],
        test_factor: float = 0.2,
        train_folds: int = 5,
        tuning_iterations: int = 20,
    ):
        BuildModelsSklearnTemplate.__init__(
            self,
            input_csv_file_name,
            target_column,
            output_file_name,
            test_factor=test_factor,
            train_folds=train_folds,
            tuning_iterations=tuning_iterations,
        )
        self.columns_to_drop = columns_to_drop
        self.positive_label = positive_label
        self.negative_label = negative_label

    def _do_at_init(self) -> None:
        self.df.drop(columns=self.columns_to_drop, inplace=True)
        self.df[self.target_column] = self.df[self.target_column].map(
            lambda x: self.positive_label if x == self.positive_label
            else self.negative_label
        )

    def _do_preprocessing(self) -> None:
        x_train, x_test \
            = scale_features(self.x_train, self.x_test, self.x_test.columns)

    def _do_print_evaluations(
        self, train_predictions: list, test_predictions: list, model_name: str
    ) -> None:
        print_stdout_and_file(
            '\t\t\taccuracy on train: '
            f'{accuracy_score(self.y_train.values, train_predictions)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tf1 on train: '
            f'{f1_score(self.y_train.values, train_predictions, pos_label=self.positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\taccuracy on test: '
            f'{accuracy_score(self.y_test.values, test_predictions)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tprecision on test: '
            f'{precision_score(self.y_test.values, test_predictions, pos_label=self.positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tprecision on test: '
            f'{precision_score(self.y_test.values, test_predictions, pos_label=self.positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\trecall on test: '
            f'{recall_score(self.y_test.values, test_predictions, pos_label=self.positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file(
            '\t\t\tf1 on test: '
            f'{f1_score(self.y_test.values, test_predictions, pos_label=self.positive_label)}',
            self.file_pointer
        )
        print_stdout_and_file("", self.file_pointer)
        plot_confusion_matrix(
            self.y_test.values,
            test_predictions,
            self.df[self.target_column].unique(),
            f"binary_{model_name}_average_confusion_matrix.png",
        )


process = BuildBinaryModelsSklearn(
    input_csv_file_name='resources/data/original/npf_train.csv',
    target_column='class4',
    positive_label='nonevent',
    negative_label='event',
    output_file_name
        ='resources/machine_learning_results/binary_classification_models.txt',
    columns_to_drop=['id', 'date', 'partlybad'],
    test_factor=0.2,
    train_folds=5,
    tuning_iterations=20,
).compute()
