# -*- coding: utf-8
import math
import os.path

import torch.utils.data
import json
import numpy as np
import preprocess


class MyDataset(torch.utils.data.Dataset):
    """
    加载自定义的数据集，也就是tokens.json
    * 请注意，该加载器会将所有数据加载至内存，如果有需要请自行修改_create_data以及get_item方法 *
    """
    # 切分数据集比率
    _ratio = 0.8

    def __init__(self, json_file_path='', train=True):
        assert os.path.exists(json_file_path), 'json file {} does not exist.'
        self.json_path = json_file_path
        self.train = train

        # vocab
        self.vocab = None

        # 生成数据集
        self.data, self.label, self.label_text = self._create_data()

    def _read_json(self):
        """
        取三个东西
        label_text = []
        tokens_all = []，总共切分的不重复个数
        tokens_per_class = { }, 每个分类下：【【tokens_of_file1】, []】
        :return:
        """
        with open(self.json_path, 'r') as f:
            json_dict = json.load(f)
        label_texts = json_dict['label_texts']
        tokens_all = json_dict['tokens_all']
        tokens_per_class = json_dict['tokens_per_class']
        return label_texts, tokens_all, tokens_per_class

    def _create_data(self):
        data, label = [], []
        label_texts, tokens_all, tokens_per_class = self._read_json()

        # 切分比率 0.8
        ratio = self._ratio
        self.vocab = preprocess.Vocab(tokens=tokens_all) # 用来将token，转换为idx
        for idx, label_name in enumerate(label_texts):
            tokens_all_files = tokens_per_class[label_name]

            num_train_per_class = len(tokens_all_files)
            train_slices = slice(0, int(math.ceil(num_train_per_class * ratio)))
            test_slices = slice(int(math.ceil(num_train_per_class * ratio)), num_train_per_class)

            # 该类别下的所有文件的tokens(tokens_per_file)
            tokens_list = tokens_all_files[train_slices] if self.train \
                                                        else tokens_all_files[test_slices]
            for tokens in tokens_list:
                row_data = [vocab.token_to_idx(token) for token in tokens]
                data.append(row_data)
            label.extend([idx for _ in range(len(tokens_list))])
        return data, label, label_texts

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return self.vocab

    def get_labels(self):
        return self.label_text

if __name__ == '__main__':
    pass