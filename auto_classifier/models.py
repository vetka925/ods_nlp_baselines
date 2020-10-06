from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import pandas as pd
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
from auto_classifier import utils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from auto_classifier.vectorizers import (
    SupervisedVectorizer,
    SupervisedStackedVectorizer,
    StackedVectorizer,
)


class BaseAutoLinearModel:
    def __init__(
        self, estimator, params, X_train, X_test, y_train, y_test, scoring_funcs_dict
    ):
        self.estimator = estimator
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scoring_func_dict = scoring_funcs_dict

        self.vectorizers = None
        self.report = None

    def vectorizers_prepare(self):
        """Create self.vectorizers -> dict()  {vec_name: vectorizer}"""
        raise NotImplemented

    def model_selection_report(self):
        assert self.vectorizers is not None, "Prepare vectorizers dict. Use vectorizers_prepare()"
        result = {"vectorizer": [], "model": [], "params": []}
        for label in self.y_train.unique():
            result[f"{label}_precision"] = []
            result[f"{label}_recall"] = []
        result.update({k: [] for k in self.scoring_func_dict})
        with tqdm(total=len(self.vectorizers)*len(self.params)) as progress_bar:
            for vec_name in self.vectorizers:
                train_features = self.vectorizers[vec_name].transform(self.X_train)
                test_features = self.vectorizers[vec_name].transform(self.X_test)
                for param in self.params:
                    clasifier = self.estimator()
                    clasifier.set_params(**param)
                    clasifier.fit(train_features, self.y_train)
                    preds = clasifier.predict(test_features)
                    classification_reports = classification_report(
                        self.y_test, preds, output_dict=True
                    )
                    for label in self.y_train.unique():
                        result[f"{label}_precision"].append(
                            classification_reports[str(label)]["precision"]
                        )
                        result[f"{label}_recall"].append(
                            classification_reports[str(label)]["recall"]
                        )
                    for score in self.scoring_func_dict:
                        metric = self.scoring_func_dict[score](self.y_test, preds)
                        result[score].append(metric)
                    result["params"].append(param)
                    result["vectorizer"].append(vec_name)
                    result["model"].append(clasifier)
                    progress_bar.set_description('Model: {}, score: {}'.format(str(clasifier), metric))
                    progress_bar.update()
        self.report = pd.DataFrame(result)
        return self.report


class AutoLinearModel(BaseAutoLinearModel):
    def vectorizers_prepare(self):
        """Create self.vectorizers -> dict() with vectorizers"""
        self.vectorizers = {}
        count_word = CountVectorizer(
            analyzer="word", ngram_range=(1, 1), max_features=15000, max_df=0.7
        )
        count_word.fit(pd.concat([self.X_train, self.X_test], ignore_index=True))
        count_char = CountVectorizer(
            analyzer="char", ngram_range=(1, 4), max_features=30000, max_df=0.7
        )
        count_char.fit(pd.concat([self.X_train, self.X_test], ignore_index=True))
        count_ngram = CountVectorizer(
            analyzer="char", ngram_range=(4, 4), max_features=30000, max_df=0.7
        )
        count_ngram.fit(pd.concat([self.X_train, self.X_test], ignore_index=True))
        tfidf_word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 1), max_features=15000, max_df=0.7
        )
        tfidf_word.fit(pd.concat([self.X_train, self.X_test], ignore_index=True))
        tfidf_char = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 4), max_features=30000, max_df=0.7
        )
        tfidf_char.fit(pd.concat([self.X_train, self.X_test], ignore_index=True))
        tfidf_ngram = TfidfVectorizer(
            analyzer="char", ngram_range=(4, 4), max_features=30000, max_df=0.7
        )
        tfidf_ngram.fit(pd.concat([self.X_train, self.X_test], ignore_index=True))
        supervised_ngram = SupervisedVectorizer(count_ngram, self.y_train)
        supervised_ngram.fit(self.X_train)
        supervised_word = SupervisedVectorizer(count_word, self.y_train)
        supervised_word.fit(self.X_train)

        self.vectorizers["tfidf_ngram_level"] = tfidf_ngram
        self.vectorizers["tfidf_word_level"] = tfidf_word
        self.vectorizers["supervised_ngram_level"] = supervised_ngram
        self.vectorizers["supervised_word_level"] = supervised_word

class BertFinetuner(pl.LightningModule):
    def __init__(self, bert_model, train_dataloader, val_dataloader, test_dataloader=None, scoring_func=f1_score):
        super(BertFinetuner, self).__init__()

        self.bert = bert_model
        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.test_data = test_dataloader
        self.classifier = torch.nn.Sequential(torch.nn.Linear(bert_model.config.hidden_size, 2),
                                              torch.nn.ReLU(inplace=True))
        self.scoring_func = scoring_func

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _, attn = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)

        h_cls = h[:, 0]
        logits = self.classifier(h_cls)
        return logits, attn

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(y_hat, label)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(y_hat, label)
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = self.scoring_func(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.val_data