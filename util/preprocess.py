# -*- coding: utf-8 -*-
import sys

import hanlp
import os
import collections
import json
from tqdm import tqdm

# extern package
tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)


def _split_text(raw_txt: str):
    return tokenizer(raw_txt)


class ReadDatasetFold:
    """
    数据集目录：
        -dir_path
            -class1: 1.txt, ...
            -class2: 2.txt, ...
            ...
    """
    
    #每个类别最多取这些数据
    _threshold = 20

    def __init__(self, dir_path='/Users/jffalbert/Workspace/pycharmProject/semantic_computation/dataset/News'):
        assert os.path.exists(dir_path), 'file {} does not exist.'.format(dir_path)
        self.dir_path = dir_path

    def tokenize_all_files(self, save_file=True):
        """
        生成目录文本列表，一个大词表tokens和每个文件的tokens
        将其写入文件
        :return: label_text, tokens_all, tokens_per_class
        """
        filter_dir = ['preprocessed']
        class_dirs = [dirname for dirname in os.listdir(self.dir_path)
                      if not dirname.endswith('json') and dirname not in filter_dir]

        # 保存label，按顺序

        print('Processing split files work....')
        label_texts = []
        tokens_all = []

        # 检查是否已经预处理过
        label_texts, tokens_all = self.check_preprocess()

        for class_dir_name in class_dirs:  # heavy work
            if class_dir_name in label_texts:
                continue

            tokens_per_text = []
            class_dir_path = os.path.join(self.dir_path, class_dir_name)
            filenames = [filename for filename in os.listdir(class_dir_path)]

            # collect tokens of every file in this class_dirname
            filenames = filenames[ : min(len(filenames), self._threshold)]           # 取有限个数
            process_bar = tqdm(filenames, file=sys.stdout)  # tqdm 是一个进度条包
            for file in process_bar:
                file_path = os.path.join(class_dir_path, file)
                with open(file_path, 'r') as f:
                    raw_text = ''.join(f.readlines())
                    tokens = _split_text(raw_text)
                    tokens_per_text.append(tokens)
                    tokens_all = tokens_all.union(set(tokens))

                # processing_bar info
                process_bar.desc = f'processing class {class_dir_name}, file {file}'

            label_texts.append(class_dir_name)

            if save_file:
                self.write_to_json_file(class_dir_name, tokens_per_text)
                # 还需要更新词表
                self.write_to_json_file('tokens_all', list(tokens_all))

    def check_preprocess(self):
        # 加载已经处理的类别名
        pre_path = os.path.join(self.dir_path, 'preprocessed')
        cls_names = [filename.split('.')[0] for filename in os.listdir(pre_path) if filename != 'tokens_all.json']

        # 判断是否存在已经处理的tokens
        tokens_exist = set()
        if os.path.exists(os.path.join(pre_path, 'tokens_all.json')):
            with open(os.path.join(pre_path, 'tokens_all.json'), 'r') as f:
                tokens_exist = set(json.load(f))

        return cls_names, tokens_exist

    def write_to_json_file(self, class_dir_name, tokens_list):
        # 将每个类别的训练数据放到对应的json文件
        json_str = json.dumps(tokens_list, indent=2)
        save_path = os.path.join(self.dir_path, 'preprocessed', f'{class_dir_name}.json')
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
        self.token_freqs = sorted(counter, key=lambda x: x[1], reverse=True)

        # unique tokens, to delete repeatable tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + \
                                            [token for token, freq in self.token_freqs if freq > min_freq])))
        self.token_to_idx = { token: idx for idx, token in enumerate(self.idx_to_token) }

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        return [ self.token_to_idx.get(token, ) for token in tokens ]

    def to_tokens(self, indices):
        return [ self.idx_to_token[idx] for idx in indices]

    @property
    def unk(self):
        return self.token_to_idx['<unk>']


def main():
    dir_path = '/home/anthony/Downloads/datasets/THUCNews'
    read_obj = ReadDatasetFold(dir_path=dir_path)
    read_obj.tokenize_all_files(save_file=True)
    print('-' * 50)


if __name__ == '__main__':
    main()



