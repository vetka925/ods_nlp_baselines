import pandas as pd
import seaborn as sns
from auto_classifier import utils
from sklearn.model_selection import train_test_split
from auto_classifier.models import AutoLinearModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


class DataProcessor:
    """
    Load, preprocess and prepare train and test data. Get base statistics.
    """

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
        self.z_scores = None

    def load_data(self, sep=","):
        if isinstance(self.data, str):
            self.loaded_data = pd.read_csv(self.data, sep=sep)[[self.class_column_name, self.text_colum_name]]
        else:
            self.loaded_data = self.data[[self.class_column_name, self.text_colum_name]]

        self.loaded_data.dropna(inplace=True)
        self.loaded_data.drop_duplicates(inplace=True)
        self.loaded_data = self.loaded_data.sample(frac=1, random_state=10).reset_index(drop=True)

        if self.test is not None:
            if isinstance(self.test, str):
                self.loaded_test = pd.read_csv(self.test)[[self.class_column_name, self.text_colum_name]]
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

        print("###### DATA INFO ######")
        print("\n### TEXT LENGTH ###\n")
        print(f"Total number of samples: {len(self.loaded_data)}")
        print(f'Classname: {str(self.class_column_name)}')
        print("\n### TEXT LENGTH ###\n")
        sns.distplot(self.text_lens, kde=False).set(title="Text length distribution")
        print(self.text_lens.describe())
        print("\n#################\n")
        if any(self.z_scores > 3):
            print("\nWARNING: OUTLIED TEXT LENGTH VALUES")
            print("TRY TO USE remove_outliers() METHOD\n")
        print("\n#################\n")

        print(f'### {self.class_column_name.upper()} ###\n')
        ind2label = {i: label for i, label in enumerate(self.loaded_data[self.class_column_name].unique())}
        print(f"Num classes: {len(ind2label)}\n")
        for i, label in ind2label.items():
            print(f"Class {i}: {label}\n")
        sns.catplot(
            x=self.class_column_name,
            kind="count",
            palette="ch:.25",
            data=self.loaded_data,
        ).set(title=f'{self.class_column_name.upper()} Label counts')
        print("\n#################\n")
        print("\n### DATA HEAD ###\n")
        print(self.loaded_data.head(10))

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


class BaseAutoClassifier:
    def __init__(self, data_processor, estimator=LinearSVC):
        assert isinstance(data_processor, DataProcessor), "data_processor must be instance of DataProcessor()"
        self.data_processor = data_processor
        if self.data_processor.X_train is None:
            if self.data_processor.loaded_data is None:
                self.data_processor.load_data()
            self.data_processor.prepare()
        self.auto_linear_model = AutoLinearModel
        self.estimator = estimator
        self.report_df = None
        self.model_params = None

    def params_prepare(self):
        """Create self.model_params -> dict()"""
        raise NotImplemented

    def get_report(self, metrics_dict={"f1": f1_score}):
        base = self.auto_linear_model(
            estimator=self.estimator,
            params=self.model_params,
            X_train=self.data_processor.X_train,
            X_test=self.data_processor.X_test,
            y_train=self.data_processor.y_train,
            y_test=self.data_processor.y_test,
            scoring_funcs_dict=metrics_dict,
        )
        base.vectorizers_prepare()
        self.vectorizers = base.vectorizers
        self.report_df = base.model_selection_report()
        return self.report_df

    def best_classifier_vectorizer(self, scoring_func_name):
        assert self.report_df is not None, "No classifaction report. Use get_report()"
        best = self.report_df.iloc[self.report_df[scoring_func_name].argmax()]
        return best.model, self.vectorizers[best.vectorizer]

    def fit_whole_data(self, estimator, vec_name, classname, ratio=None):
        all_texts = pd.concat([self.data_processor.X_train, self.data_processor.X_test])
        all_labels = pd.concat([self.data_processor.y_train[classname], self.data_processor.y_test[classname]])
        if ratio:
            X_train, y_train = utils.balance_binary_data(all_texts, all_labels, ratio=ratio)
        else:
            X_train, y_train = all_texts, all_labels
        features = self.vectorizers[vec_name].transform(X_train)
        estimator.fit(features, y_train)
        return estimator

class AutoClassifier(BaseAutoClassifier):

   def params_prepare(self):
        C = [0.01, 0.1, 1, 10, 100]
        tol = [1e-3, 1e-4, 1e-5]
        loss = ["hinge", "squared_hinge"]
        self.model_params = []
        for c in C:
            for l in loss:
                for t in tol:
                    self.model_params.append({"C": c, "loss": l, "tol": t})