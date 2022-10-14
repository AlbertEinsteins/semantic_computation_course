# -*- coding: utf-8 -*-
import sys

import hanlp
import os
import collections
import json
from tqdm import tqdm

# extern package
tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)


class ReadDatasetFold:
    """
    数据集目录：
        -dir_path
            -class1: 1.txt, ...
            -class2: 2.txt, ...
            ...
    """
    def __init__(self, dir_path='/Users/jffalbert/Workspace/pycharmProject/semantic_computation/dataset/News'):
        assert os.path.exists(dir_path), 'file {} does not exist.'.format(dir_path)
        self.dir_path = dir_path

    def tokenize_all_files(self, save_file=None):
        """
        生成目录文本列表，一个大词表tokens和每个文件的tokens
        将其写入文件
        :return: label_text, tokens_all, tokens_per_class
        """
        class_dirs = [dirname for dirname in os.listdir(self.dir_path)]

        # 得到总共的tokens，去重
        tokens_all = set()
        # 收集每个分类，每篇文章的tokens，不去重
        # { 'label1': [[tokens_file1], [tokens_file2], ...], 'label2': [[]] }
        tokens_per_class = {}
        # 保存label，按顺序
        label_texts = []

        print('Processing split files work....')
        for class_dir_name in class_dirs:  # heavy work
            tokens_per_text = []
            class_dir_path = os.path.join(self.dir_path, class_dir_name)
            filenames = [filename for filename in os.listdir(class_dir_path)]

            # collect tokens of every file in this class_dirname
            process_bar = tqdm(filenames, file=sys.stdout)  # tqdm 是一个进度条包
            for file in process_bar:
                file_path = os.path.join(class_dir_path, file)
                with open(file_path, 'r') as f:
                    raw_text = ''.join(f.readlines())
                    tokens = self._split_text(raw_text)
                    tokens_per_text.append(tokens)
                    tokens_all = tokens_all.union(set(tokens))

                # processing_bar info
                process_bar.desc = f'processing class {class_dir_name}, file {file}'

            label_texts.append(class_dir_name)
            tokens_per_class[class_dir_name] = tokens_per_text

        # labels, tokens_all，及tokens_per_text保存起来
        save_dict = { "label_texts": label_texts, "tokens_all": tokens_all, "tokens_per_text": tokens_per_class }
        if save_file is not None:
            save_path = os.path.abspath(save_file)
            json_str = json.dumps(save_dict, indent=2)
            with open(save_path, 'w') as f:
                f.write(json_str)

        return label_texts, tokens_all, tokens_per_class

    def _split_text(self, raw_txt: str):
        return tokenizer(raw_txt)


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

def test():
    a = { 'testxxx': '123', "test": '12312' }
    json_str = json.dumps(a, indent=2)
    with open('test.json', 'w') as f:
        f.write(json_str)


def main():
    read_obj = ReadDatasetFold()
    save_file = os.path.join(os.getcwd(), 'tokens.json')
    label_text, tokens_all, _ = read_obj.tokenize_all_files(save_file=save_file)
    print('-' * 50)
    print(label_text)
    print(len(tokens_all))


if __name__ == '__main__':
    main()

