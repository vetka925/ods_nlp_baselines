import pandas as pd
import seaborn as sns
import utils
from sklearn.model_selection import train_test_split
from models import AutoLinearModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


class DataProcessor:
    def __init__(self, data, test=None, text_colum_name=None, class_column_name=None):
        self.data = data
        self.test = test
        self.text_colum_name = text_colum_name if text_colum_name else "text"
        self.class_column_name = class_column_name if class_column_name else "label"

        self.loaded_data = None
        self.loaded_test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.text_lens = None
        self.mean_text_len = None
        self.ind2label = None
        self.z_scores = None
        self.label2ind = None

    def load_data(self, sep=","):
        if isinstance(self.data, str):
            self.loaded_data = pd.read_csv(self.data, sep=sep)[
                [self.text_colum_name, self.class_column_name]
            ]
        else:
            self.loaded_data = self.data[[self.text_colum_name, self.class_column_name]]

        self.loaded_data.dropna(inplace=True)
        self.loaded_data.drop_duplicates(inplace=True)

        if self.test is not None:
            if isinstance(self.test, str):
                self.loaded_test = pd.read_csv(self.test)[
                    [self.text_colum_name, self.class_column_name]
                ]
            else:
                self.loaded_test = self.test

            self.loaded_test.dropna(inplace=True)
            self.loaded_test.drop_duplicates(inplace=True)

        self.text_lens = self.loaded_data[self.text_colum_name].apply(
            lambda text: len(text)
        )
        self.mean_text_len = self.text_lens.mean()
        self.z_scores = (
            (self.text_lens - self.mean_text_len) / self.text_lens.std(ddof=0)
        ).abs()
        self.ind2label = {
            i: label
            for i, label in enumerate(self.loaded_data[self.class_column_name].unique())
        }
        self.label2ind = {l: i for i, l in self.ind2label.items()}

        print("###### DATA INFO ######")
        print("\n### TEXT LENGTH ###\n")
        sns.distplot(self.text_lens, kde=False).set(title="Text length distribution")
        print(self.text_lens.describe())
        print("\n#################\n")
        if any(self.z_scores > 3):
            print("\nWARNING: OUTLIED TEXT LENGTH VALUES")
            print("TRY TO USE remove_outliers() METHOD\n")
        print("\n#################\n")
        print(f"Num classes: {len(self.ind2label)}\n")
        for i, label in self.ind2label.items():
            print(f"Class {i}: {label}\n")
        print("\n#################\n")
        print("\n### DATA HEAD ###\n")
        print(self.loaded_data.head(10))
        sns.catplot(
            x=self.class_column_name,
            kind="count",
            palette="ch:.25",
            data=self.loaded_data,
        ).set(title="Label counts")
        return "Data loaded"

    def remove_outliers(self):
        assert self.loaded_data is not None, "Data is not loaded"

        self.loaded_data = self.loaded_data[self.z_scores < 3]
        print("###### DATA INFO ######")
        print("\n### TEXT LENGTH ###\n")
        sns.distplot(self.text_lens, kde=False).set(
            title="Text length distribution without outliers"
        )
        print(self.text_lens.describe())
        return "Outliers removed"

    def prepare(self, clean_regexp_patterns=None, test_size=0.15):
        assert self.loaded_data is not None, "Data is not loaded"

        self.loaded_data[self.text_colum_name] = [
            utils.clean(text, clean_regexp_patterns)
            for text in self.loaded_data[self.text_colum_name]
        ]

        if self.test is not None:
            self.loaded_test[self.text_colum_name] = [
                utils.clean(text, clean_regexp_patterns)
                for text in self.loaded_test[self.text_colum_name]
            ]
            self.X_train = self.loaded_data[self.text_colum_name]
            self.y_train = self.loaded_data[self.class_column_name]
            self.X_test = self.loaded_test[self.text_colum_name]
            self.y_test = self.loaded_test[self.class_column_name]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.loaded_data[self.text_colum_name],
                self.loaded_data[self.class_column_name],
                test_size=test_size,
                random_state=10,
            )
        sns.catplot(
            x=self.class_column_name,
            kind="count",
            palette="ch:.25",
            data=pd.DataFrame(self.y_train),
        ).set(title="Train Label counts")
        sns.catplot(
            x=self.class_column_name,
            kind="count",
            palette="ch:.25",
            data=pd.DataFrame(self.y_test),
        ).set(title="Test Label counts")

        return "Data prepared for train"


class AutoClassifier:
    def __init__(self, data_processor):
        assert isinstance(data_processor, DataProcessor), "data_processor must be instance of DataProcessor()"
        self.data_processor = data_processor
        if self.data_processor.X_train is None:
            if self.data_processor.loaded_data is None:
                self.data_processor.load_data()
            self.data_processor.prepare()

    def params_prepare(self):
        C = [0.01, 0.1, 1, 10, 100]
        solver = ["liblinear", "sag", "lbfgs"]
        tol = [1e-3, 1e-4, 1e-5]
        loss = ["hinge", "squared_hinge"]

        self.logreg_params = []
        for c in C:
            for s in solver:
                for t in tol:
                    self.logreg_params.append({"C": c, "solver": s, "tol": t})

        self.svm_params = []
        for c in C:
            for l in loss:
                for t in tol:
                    self.svm_params.append({"C": c, "loss": l, "tol": t})

    def get_report(self, metrics_dict={"f1": f1_score}):
        reports = []
        logregs = AutoLinearModel(
            LogisticRegression,
            self.logreg_params,
            self.data_processor.X_train,
            self.data_processor.X_test,
            self.data_processor.y_train,
            self.data_processor.y_test,
            metrics_dict,
        )
        logregs.vectorizers_prepare()
        svms = AutoLinearModel(
            LinearSVC,
            self.svm_params,
            self.data_processor.X_train,
            self.data_processor.X_test,
            self.data_processor.y_train,
            self.data_processor.y_test,
            metrics_dict,
        )
        svms.vectorizers_prepare()
        self.vectorizers = logregs.vectorizers
        reports.append(logregs.model_selection_report())
        reports.append(svms.model_selection_report())
        return pd.concat(reports, ignore_index=True)
