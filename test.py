# -*- coding: utf-8 -*-
import os.path
import sys

import torch
import torch.utils.data
from tqdm import tqdm

from model import lstm
from util import dataloader, log

configs = {
    "lr": 8e-3,
    "epochs": 5,
}
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def collate_fn(batch):
    max_len = 500
    (x_list, y_list) = zip(*batch)

    # 长了，截断；短了，补0
    pad_f = lambda x: x[:max_len] if len(x) >= max_len \
                                else x + [0 for _ in range(max_len - len(x))]
    x_list = list(map(pad_f, x_list))
    X = torch.stack([torch.tensor(x, dtype=torch.int32) \
                     for x in x_list], dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y


def load_data(json_path='', tokens_filepath=''):
    test_data = dataloader.MyDataset(json_filepath=json_path, tokens_path=tokens_filepath, train=False)
    return test_data


def data_loader(data, batch_size=64, shuffle=True, num_workers=0):
    return torch.utils.data.DataLoader(data, batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       collate_fn=collate_fn,
                                       )
def main():
    device = get_device()
    # ===================== prepare training =====================
    # load data
    json_path = '/home/xf/disk/dataset/THUCNews/data.json'
    assert os.path.exists(json_path), 'dataset json file {} does not exist.'.format(json_path)

    tokens_path = '/home/xf/disk/dataset/THUCNews/tokens_all.txt'
    assert os.path.exists(tokens_path), 'tokens json file {} does not exist.'.format(tokens_path)

    test_data = load_data(json_path, tokens_path)

    test_iter = data_loader(test_data)
    # set network
    label_text = test_data.get_labels()
    vocab = test_data.get_vocab()

    vocab_size = len(vocab)
    num_embeddings = 100
    num_hiddens = 64
    num_classes = len(label_text)
    net = lstm.Net(vocab_size=vocab_size, embed_size=num_embeddings, num_hiddens=num_hiddens,
                   num_classes=num_classes, device=get_device())
    net = net.to(device)

    # ========================= testing =======================
    val_nums = len(test_data)
    save_weight_path = os.path.join(os.getcwd(), 'pretrained', 'pretrained.pth')

    # Load weight
    if os.path.exists(save_weight_path):
        net.load_state_dict(torch.load(save_weight_path, map_location=lambda storage, loc: storage.cuda(0)))

        # 校验验证集的准确率
    net.eval()
    correct_num = 0
    with torch.no_grad():
        val_bar = tqdm(test_iter, file=sys.stdout)
        for step, (X, y) in enumerate(val_bar):
            X, y = X.to(device), y.to(device)
            logits = net(X)
            pred = torch.argmax(logits, dim=-1)
            correct_num += torch.eq(pred, y).sum().item()
            val_bar.desc = 'step: [{}/{}], accuracy: {}'.format(step, len(test_iter),
                                                                torch.eq(pred, y).sum().item() / len(y))
    val_acc = correct_num / val_nums
    print('testing accuracy, {}'.format(val_acc))


if __name__ == '__main__':
    main()
