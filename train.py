# -*- coding: utf-8 -*-
import os.path
import sys

import torch
import torch.utils.data
from torch import optim, nn
from tqdm import tqdm

from model import lstm
from util import dataloader

configs = {
    "lr": 1e-4,
    "epochs": 10,
}


def load_data(json_path=''):
    train_data, test_data = dataloader.MyDataset(json_file_path=json_path, train=True), \
                            dataloader.MyDataset(json_file_path=json_path, train=False)
    return train_data, test_data


def data_loader(data, batch_size=16, shuffle=True, num_workers=4):
    return torch.utils.data.DataLoader(data, batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)



def main():
    # ===================== prepare training =====================
    # load data
    json_path = '/home/anthony/Downloads/datasets/THUCNews/tokens.json'
    assert os.path.exists(json_path), 'dataset json file {} does not exist.'.format(json_path)
    train_data, test_data = load_data(json_path)

    train_iter = data_loader(train_data)
    test_iter = data_loader(test_data)

    # set network
    label_text = train_data.get_labels()
    vocab = train_data.get_vocab()

    vocab_size = len(vocab)
    num_embeddings = 100
    num_hiddens = 64
    num_classes = len(label_text)
    net = lstm.Net(vocab_size=vocab_size, embed_size=num_embeddings, num_hiddens=num_hiddens,
                   num_classes=num_classes)

    # set optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params=net.parameters(), lr=configs['lr'])
    # set loss function
    loss_fun = nn.CrossEntropyLoss()

    # ========================= training =======================
    best_acc = 0.0
    val_nums = len(test_data)
    epochs = configs['epochs']
    train_steps = len(train_iter)
    save_weight_path = ''
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_iter, file=sys.stdout)

        for (X, y) in train_bar:
            optimizer.zero_grad()
            logits = net(X)
            loss = loss_fun(logits, y)
            loss.backward()
            optimizer.step()

            # 打印，并保存到文件中
            running_loss += loss
            train_bar.desc = f'training epoch [{epoch}/{epochs}], loss: {loss:.3f}'

        # 校验验证集的准确率
        net.eval()
        correct_num = 0
        with torch.no_grad():
            val_bar = tqdm(test_iter, file=sys.stdout)
            for X, y in val_bar:
                logits = net(X)
                correct_num += torch.eq(logits, y).sum().item()
        val_acc = correct_num / val_nums
        # 存储权重
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_weight_path)


if __name__ == '__main__':
    main()
