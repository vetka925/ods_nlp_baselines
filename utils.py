import re
from torch.utils.data import TensorDataset, random_split

RE_URL = re.compile(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S")

def clean(text, patterns=None):
    text = RE_URL.sub("", text)
    text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "", text)
    if patterns:
        for pattern in patterns:
            text = re.sub(pattern, "", text)
    return text.lower()

def prepare_features(text, tokenizer):
    encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True,
                            max_length = 128,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            truncation=True,
                            return_tensors = 'pt',
                      )
    return encoded_dict['input_ids'], encoded_dict['attention_mask'], encoded_dict['token_type_ids']

def prepare_dataset(texts, labels, tokenizer, val_size=None):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True,
                            max_length = 128,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            truncation=True,
                            return_tensors = 'pt',
                      )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])

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

def predict_bert(texts, bert_model, tokenizer):
    predictions = []
    bert_model.eval()
    for t in texts:
      tok, atten, ttype = prepare_features(t, tokenizer)
      predictions.append(int(torch.argmax(bert_finetuner(tok, atten, ttype)[0].detach().cpu())))
    return predictions