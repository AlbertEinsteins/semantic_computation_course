# -*- coding: utf-8
import math
import os.path

import torch.utils.data
import json
from . import preprocess


class MyDataset(torch.utils.data.Dataset):
    """
    从磁盘上加载数据
    """
    # 切分数据集比率
    _ratio = 0.8

    def __init__(self, json_filepath='', tokens_path='', train=True):
        assert os.path.exists(json_filepath), 'json directory {} does not exist.'.format(json_filepath)
        self.json_filepath = json_filepath
        self.tokens_path = tokens_path
        self.train = train

        # 读取数据路径索引
        self.data, self.labels, self.label_texts, tokens_all = self._read_json()
        # vocab
        self.vocab = preprocess.Vocab(tokens=tokens_all)

    def _read_json(self, is_train=True):
        """
        取三个东西
        label_text = []
        data = []
        labels = []
        :return:
        """
        ratio = self._ratio

        with open(self.json_filepath, 'r') as f:
            data_dict = json.load(f)
        label_texts = data_dict['label_texts']
        labels = data_dict['labels']
        data = data_dict['data']

        num_len = len(data)
        train_slices = slice(0, math.floor(ratio * num_len))
        test_slices = slice(math.ceil(ratio * num_len), num_len)

        data = data[train_slices] if is_train else data[test_slices]
        labels = labels[train_slices] if is_train else data[test_slices]

        tokens_all = []
        with open(self.tokens_path, 'r') as f:
            tokens_all.extend(list(f.readline()))

        return data, labels, label_texts, tokens_all

    def __getitem__(self, idx):
        data_path = self.data[idx]
        label = self.labels[idx]

        with open(data_path, 'r') as f:
            raw_text = ''.join(f.readlines())
        tokens = preprocess.split_text(raw_text)
        X = self.vocab[tokens]
        return X, label

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return self.vocab

    def get_labels(self):
        return self.label_texts


if __name__ == '__main__':
    pass
