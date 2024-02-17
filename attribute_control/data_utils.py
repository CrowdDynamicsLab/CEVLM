import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SpeedDataset(Dataset):
    def __init__(
        self,
        sentences_with_speeds,
        tokenizer,
        transform = None,
        predict_difference: bool = False,
        switch = None,
        max_len: int = 512,
    ):
        self.sentences_with_speeds = sentences_with_speeds
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.predict_difference = predict_difference
        self.transform = transform
        self.switch = switch

    def __len__(self):
        return len(self.sentences_with_speeds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx.data)

        data_idx = self.sentences_with_speeds[idx]
        text = data_idx[0]
        speed = float(data_idx[-1])

        # TODO: temp fix, inverted features
        if len(data_idx) == 5:
            speed *= -1

        label = torch.tensor(speed, dtype=torch.float) if not self.switch else torch.tensor(self.switch[speed], dtype=torch.int)

        item = {
            'text': text if not self.predict_difference \
                    else text + ' ' + data_idx[1],  # two sentences used when predicting delta,
            'label': label
        }

        if self.transform:
            item = self.transform(item)
        else:
            item = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                # padding='max_length',
                return_token_type_ids=True,
                return_tensors='pt'
            )

            del item['token_type_ids']

            # truncation when text is too long
            # for some reason, max_length doesn't work 
            for k, v in item.items():
                if v.shape[1] > self.max_len:
                    item[k] = v[:, :self.max_len]
                item[k] = item[k].to(torch.long).squeeze(0)
        
        return item

def get_dataset(data_dir):

    with open(data_dir, 'r') as f:
        data = f.readlines()

    data = list(
        map(
            lambda x: x.strip().split('\t'),
            data
        )
    )

    return data

# Transforms
class ShuffleChunkTransform(object):
    """Shuffle the chunks of text for speed computation"""

    def __init__(self, chunk_len=3):
        self.chunk_len = chunk_len

    def chunk(self, l, n):
        n = max(1, n)
        return list(l[i:i+n] for i in range(0, len(l), n))

    def __call__(self, sample):
        text = sample['text'].split()

        text = self.chunk(text, self.chunk_len)
        random.shuffle(text)
        text = [item for sublist in text for item in sublist] # flatten

        sample['text'] = ' '.join(text)
        return sample


class TokenizeTransform(object):
    """Tokenize text of dataset"""

    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, sample):
        item = self.tokenizer(
            sample['text'],
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # truncation when text is too long
        # for some reason, max_length doesn't work 
        for k, v in item.items():
            if v.shape[1] > self.max_len:
                item[k] = v[:, :self.max_len]
            sample[k] = item[k].to(torch.long).squeeze(0)

        return sample
