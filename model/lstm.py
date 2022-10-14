# -*- coding: utf-8
import this

import torch
from torch import nn


# Init Params
def init_weights(shape, sigma=0.01):
    return nn.Parameter(torch.randn(shape) * sigma)


class LSTM(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        super(LSTM, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_inputs = num_inputs

        # Forget Gate
        self.W_xf, self.W_hf, self.b_f = self.triple()
        # Input Gate
        self.W_xi, self.W_hi, self.b_i = self.triple()
        # Input Node
        self.W_xc, self.W_hc, self.b_c = self.triple()
        # Output Gate
        self.W_xo, self.W_ho, self.b_o = self.triple()

    def forward(self, inputs):
        # inputs [序列长度, 批量大小, 词向量长度]
        batch_size = inputs.shape[1]
        H, C = self.begin_state(batch_size)
        outputs = []

        for x in inputs:
            I = torch.sigmoid((x @ self.W_xi) + (H @ self.W_hi) + self.b_i)
            F = torch.sigmoid((x @ self.W_xf) + (H @ self.W_hf) + self.b_f)
            O = torch.sigmoid((x @ self.W_xo) + (H @ self.W_ho) + self.b_o)
            C_tilda = torch.tanh((x @ self.W_xc) + (H @ self.W_hc) + self.b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            outputs.append(H)
        return outputs, (H, C)

    def begin_state(self, batch_size, device='cpu'):
        return (torch.zeros((batch_size, self.num_hiddens), device=device),
                torch.zeros((batch_size, self.num_hiddens), device=device))

    def triple(self):
        return (init_weights((self.num_inputs, self.num_hiddens)),
                init_weights((self.num_hiddens, self.num_hiddens)),
                init_weights((self.num_hiddens, ))
                )


class AdditiveAttention(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout):
        super(AdditiveAttention, self).__init__()
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

        # store
        self.attention_weights = None

    def forward(self, queries, keys, values):
        queries, keys = self.W_q(queries), self.W_k(keys)

        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 不同的query会得到不同的权重
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        # 移除hidden维度
        scores = self.w_v(features).squeeze(-1)
        # scores [batch_size, no. of queries, 键值对个数(<k, v>的个数，这里是hiddens个数)]
        self.attention_weights = self._softmax(scores)
        # values (batch_size, 键值对个数, value_size)
        return torch.bmm(self.dropout(self.attention_weights), values)

    def _softmax(self, scores):
        return nn.functional.softmax(scores, dim=-1)


class Net(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_classes=5):
        super(Net, self).__init__()
        self.num_hiddens = num_hiddens

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm_layer = LSTM(embed_size, num_hiddens)
        self.attention = AdditiveAttention(key_size=num_hiddens,
                                           query_size=num_hiddens, num_hiddens=num_hiddens, dropout=0.1)

        # 分类器
        self.classifier = nn.Linear(num_hiddens, num_classes)

    def forward(self, X):
        # X [batch_size, num_steps, embedding]
        batch_size, num_steps = X.shape
        X = self.embedding(X)
        X_swap = X.permute(1, 0, 2)
        outputs, (H, _) = self.lstm_layer(X_swap)

        # attention返回为[batch, no. of queries, value的维度]
        # [batch, hiddens] -> [batch, num_steps, hiddens]
        outputs = torch.cat(outputs, dim=-1).reshape((batch_size, num_steps, self.num_hiddens))
        H = torch.unsqueeze(H, dim=1)
        # out 为一个特征向量，表示这个文本， 通过最后时间步的H向量作为query，key和value所有时间步的输出
        out = self.attention(H, outputs, outputs).squeeze(1)
        out = self.classifier(out)
        return out



# ===========================================================================
def test1():
    batch_size, num_steps = 2, 35

    X = torch.randn((num_steps, batch_size, 80))
    net = LSTM(80, 64)
    state = net.begin_state(X.shape[1])
    Y, new_state = net(X, state)
    print(len(Y), Y[0].shape)
    print(new_state[0].shape, new_state[1].shape)


def test2():
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)

    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values).shape)


def test3():
    net = Net(vocab_size=10, embed_size=8, num_hiddens=32, num_classes=5)
    X = torch.zeros((4, 7), dtype=torch.long)
    print(net(X).shape)


if __name__ == '__main__':
    test3()
    pass
