import re
from torch.utils.data import TensorDataset, random_split
import torch

RE_URL = re.compile(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S")
RE_IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

def clean(text, patterns=None):
    text = RE_URL.sub("", text)
    text = RE_IP.sub("", text)
    if patterns:
        for pattern in patterns:
            text = re.sub(pattern, "", text)
    return text.lower()

def balance_binary_data(texts, labels, seed=10, ratio=1):
    random.seed(seed)
    unique_labels = labels.unique()
    inds0 = list(texts[labels == unique_labels[0]].index)
    inds1 = list(texts[labels == unique_labels[1]].index)
    len_dict = {len(inds0): inds0,
                len(inds1): inds1}
    max_len = max(len_dict.keys())
    min_len = min(len_dict.keys())
    if (max_len / min_len) <= ratio:
        return None, None
    samples = random.sample(len_dict[max_len], k=int(ratio * min_len))
    samples.extend(len_dict[min_len])

    X = texts.loc[samples].reset_index(drop=True)
    y = labels.loc[samples].reset_index(drop=True)
    return X, y

def prepare_features(text, tokenizer):
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True,
        return_tensors="pt",
    )
    return (
        encoded_dict["input_ids"],
        encoded_dict["attention_mask"],
        encoded_dict["token_type_ids"],
    )


def prepare_dataset(texts, labels, tokenizer, val_size=None):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])
        token_type_ids.append(encoded_dict["token_type_ids"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
    if val_size:
        val_size = int(val_size * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset
    else:
        return dataset


def predict_bert(texts, bert_finetuner, tokenizer):
    predictions = []
    bert_finetuner.eval()
    for t in texts:
        tok, atten, ttype = prepare_features(t, tokenizer)
        predictions.append(
            int(torch.argmax(bert_finetuner(tok, atten, ttype)[0].detach().cpu()))
        )
    return predictions
