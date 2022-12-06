# -*- coding: utf-8
import sys

import math
import os.path
import ast
import tqdm
import numpy as np

import torch.utils.data
import json
from . import preprocess

class MyDataset(torch.utils.data.Dataset):
    """
    从磁盘上加载数据
    """
    # 切分数据集比率
    _ratio = 0.9
    _vocab = None

    def __init__(self, json_filepath='', tokens_path='', train=True):
        assert os.path.exists(json_filepath), 'json directory {} does not exist.'.format(json_filepath)

        self.json_filepath = json_filepath
        self.tokens_path = tokens_path
        self.train = train

        # 读取数据路径索引
        self.data, self.labels, self.label_texts = self._read_json()
        # vocab
        if MyDataset._vocab is None:
            tokens_all = self.read_tokens()
            MyDataset._vocab = preprocess.Vocab(tokens=tokens_all)

    def _read_json(self):
        """
        取三个东西
        label_text = []
        data = []
        labels = []
        :return:
        """
        ratio = self._ratio
        np.random.seed(0)

        with open(self.json_filepath, 'r') as f:
            data_dict = json.load(f)
        label_texts = data_dict['label_texts']
        labels = data_dict['labels']
        data = data_dict['data']

        num_len = len(data)


        # 获取打乱后的下标
        shuffled_index = np.random.permutation(num_len)
        train_slices = shuffled_index[slice(0, math.floor(ratio * num_len))].tolist()
        test_slices = shuffled_index[slice(math.ceil(ratio * num_len), num_len)].tolist()

        data = np.array(data)
        labels = np.array(labels)
        data = data[train_slices] if self.train else data[test_slices]
        labels = labels[train_slices] if self.train else labels[test_slices]

        return data.tolist(), labels.tolist(), label_texts

    def read_tokens(self):
        tokens_all = []
        with open(self.tokens_path, 'r') as f:
            tokens_per_class = f.readlines()
        # 单线程提取
        print('提取tokens文件')
        process_bar = tqdm.tqdm(tokens_per_class, file=sys.stdout)
        for ind, line in enumerate(process_bar):
            process_bar.desc = 'extract tokens in file {}/{}'.format(ind + 1, len(tokens_per_class))
            tokens_list = ast.literal_eval(line)
            tokens_all.extend(tokens_list)
            del tokens_list

        return tokens_all

    def __getitem__(self, idx):
        data_path = self.data[idx]
        label = self.labels[idx]

        with open(data_path, 'r') as f:
            raw_text = ''.join(f.readlines())
        tokens = preprocess.split_text(raw_text)
        X = MyDataset._vocab[tokens]
        return X, label

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return MyDataset._vocab

    def get_labels(self):
        return self.label_texts


if __name__ == '__main__':
    pass
