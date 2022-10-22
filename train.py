# -*- coding: utf-8 -*-
import os.path
import sys

import torch
import torch.utils.data
from torch import optim, nn
from tqdm import tqdm

from model import lstm
from util import dataloader, log

configs = {
    "lr": 1e-4,
    "epochs": 5,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    train_data, test_data = dataloader.MyDataset(json_filepath=json_path, tokens_path=tokens_filepath, train=True), \
                            dataloader.MyDataset(json_filepath=json_path, tokens_path=tokens_filepath, train=False)
    return train_data, test_data


def data_loader(data, batch_size=64, shuffle=True, num_workers=4):
    return torch.utils.data.DataLoader(data, batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       collate_fn=collate_fn,
                                       )
def main():
    train_logger = log.FileLog('train.txt')
    acc_logger = log.FileLog('test.txt')
    # ===================== prepare training =====================
    # load data
    json_path = '/home/xf/disk/dataset/THUCNews/data.json'
    assert os.path.exists(json_path), 'dataset json file {} does not exist.'.format(json_path)

    tokens_path = '/home/xf/disk/dataset/THUCNews/tokens_all.txt'
    assert os.path.exists(tokens_path), 'tokens json file {} does not exist.'.format(tokens_path)

    train_data, test_data = load_data(json_path, tokens_path)

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
    net = net.to(device)

    # set optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params=params, lr=configs['lr'])
    # set loss function
    loss_fun = nn.CrossEntropyLoss()

    # ========================= training =======================
    best_acc = 0.0
    val_nums = len(test_iter)
    train_nums = len(train_iter)
    epochs = configs['epochs']
    save_weight_path = os.path.join(os.getcwd(), 'pretrained', 'pretrained.pth')

    if os.path.exists(save_weight_path):
        net.load_state_dict(torch.load(save_weight_path, device=0))
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_iter, file=sys.stdout)
        for step, (X, y) in enumerate(train_bar):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = net(X)
            loss = loss_fun(logits, y)
            loss.backward()
            optimizer.step()

            # 打印进度
            running_loss += loss
            train_bar.desc = f'training epoch [{epoch}/{epochs}], loss: {loss:.3f}'
            train_logger.log('train|epoch:{}\tstep:{}/{}\tloss:{:.4f}'.format(epoch, step, train_nums, running_loss))

        # 校验验证集的准确率
        net.eval()
        correct_num = 0
        with torch.no_grad():
            val_bar = tqdm(test_iter, file=sys.stdout)
            for X, y in val_bar:
                logits = net(X)
                pred = torch.argmax(logits, dim=-1)
                correct_num += torch.eq(pred, y).sum().item()
        val_acc = correct_num / val_nums
        print('epoch {}, training loss: {:.4f}, testing accuracy, {}'.format(epoch, running_loss / train_nums, val_acc))
        acc_logger.log('test|epoch:{}\tloss:{:.4f},acc:{:3f}'.format(epoch, running_loss / train_nums, val_acc))

        # 存储权重
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_weight_path)


if __name__ == '__main__':
    main()
