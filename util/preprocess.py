# -*- coding: utf-8 -*-
import sys

import hanlp
import os
import collections
import json
import math
from tqdm import tqdm

# extern package
tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)

def split_text(raw_txt: str):
    return tokenizer(raw_txt, )


class ReadDatasetFold:
    """
    数据集目录：
        -dir_path
            -class1: 1.txt, ...
            -class2: 2.txt, ...
            ...
    """
    
    #每个类别最多取这些数据
    _threshold = 20000

    # 每个类别取 0.9作为train, 0.1作为test
    _rate = 0.9
    def __init__(self, dir_path='/Users/jffalbert/Workspace/pycharmProject/semantic_computation/dataset/News'):
        assert os.path.exists(dir_path), 'file {} does not exist.'.format(dir_path)
        self.dir_path = dir_path

    def preprocess_data(self):
        """
        生成json文件，格式为{ 'data: [path1, path2, ...],
                            'label': [],
                            ''label_text: []
                        }
        生成tokens_all.txt, 每个类别一行[]
        """
        class_dirs = [dirname for dirname in os.listdir(self.dir_path)
                      if not dirname.endswith('json') and dirname != 'tokens_all.txt']

        # 保存label，按顺序
        print('Processing split files work....')
        label_texts = []
        data_train = []
        labels_train = []
        data_test = []
        labels_test = []

        for idx, class_dir_name in enumerate(class_dirs):  # heavy work
            class_dir_path = os.path.join(self.dir_path, class_dir_name)
            filenames = [filename for filename in os.listdir(class_dir_path)]
            # truncated
            filenames = filenames[ : min(len(filenames), self._threshold)]           # 取有限个数

            # 切分数据 9:1
            train_files = filenames[ : math.floor(0.9 * len(filenames))]
            test_files = filenames[math.ceil(0.9 * len(filenames)) : ]

            data_train.extend([os.path.join(class_dir_path, filename) for filename in train_files])
            labels_train.extend([idx for _ in range(len(train_files))])

            data_test.extend([os.path.join(class_dir_path, filename) for filename in test_files])
            labels_test.extend([idx for _ in range(len(test_files))])

            label_texts.append(class_dir_name)

            # 获取所有文件总的tokens, 很耗时间
            # process_bar = tqdm(filenames, file=sys.stdout)
            # for filename in process_bar:
            #     with open(os.path.join(class_dir_path, filename), 'r') as f:
            #         raw_text = ''.join(f.readlines())
            #     process_bar.desc = 'processing class {}, filename {}'.format(class_dir_name, filename)
            #     tokens_list = split_text(raw_text)
            #     with open(os.path.join(self.dir_path, 'tokens_all.txt'), 'a') as f:
            #         f.write(str(tokens_list) + '\n')

        # 写入文件
        data_train_dict = { 'label_texts': label_texts, 'data': data_train, 'labels': labels_train }
        data_test_dict = { 'label_texts': label_texts, 'data': data_test, 'labels': labels_test }
        self.write_to_json_file(data_train_dict, 'train.json')
        self.write_to_json_file(data_test_dict, 'test.json')

    def write_to_json_file(self, data_dict, json_filename='data.json'):
        # 将数据目录放到json文件
        json_str = json.dumps(data_dict, indent=2)
        save_path = os.path.join(self.dir_path, json_filename)
        with open(save_path, 'w') as f:
            f.write(json_str)


class Vocab:
    """提供从token <-> id的映射"""
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        if tokens and isinstance(tokens[0], list): # 如果是list，摊开
            tokens = [token for line in tokens for token in line]
        # compute freq
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # unique tokens, to delete repeatable tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + \
                                            [token for token, freq in self.token_freqs if freq > min_freq])))
        self.token_to_idx = { token: idx for idx, token in enumerate(self.idx_to_token) }

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        return [ self.token_to_idx.get(token, 0) for token in tokens ]

    def to_tokens(self, indices):
        return [ self.idx_to_token[idx] for idx in indices]

    @property
    def unk(self):
        return self.token_to_idx['<unk>']


def main():
    dir_path = '/home/xf/disk/dataset/THUCNews'
    read_obj = ReadDatasetFold(dir_path=dir_path)
    read_obj.preprocess_data()
    print('-' * 50)
    print('处理完成.')


if __name__ == '__main__':
    main()
