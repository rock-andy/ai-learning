import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        # d_model: 代表词嵌入的维度
        # dropout: Dropout的零比率
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # 位置编码矩阵 大小是 max_len * d_model
        pe = torch.zeros(max_len, d_model)

        # 绝对位置矩阵

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))

        # 将前面定义的变换矩阵 进行奇数 偶数的分别赋值

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: 文本序列的词嵌入表示

        # 明确，pe的编码天长了
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        return self.dropout(x)


d_model = 512
dropout = 0.1
maxlen = 60
vocab = 1000

x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 22]]))

emb = Embeddings(d_model, vocab)
embr = emb(x)
print("embr:", embr)
print(embr.shape)

x = embr
pe = PositionalEncoding(d_model, dropout, maxlen)

pe_result = pe(x)

print(pe_result)
print(pe_result.shape)


def subsequent_mask(size):
    attn_shap = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shap), k=1).astype('uint8')

    # 使得三角矩阵翻转

    return torch.from_numpy(1-subsequent_mask)


size = 5

sm = subsequent_mask(size)
print('##############')
print(sm)


def attention(query, key, value, mask=None, dropout=None):
    # query: key, value 三个张量
    # mask: z
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    
    return torch.matmul(p_attn, value), p_attn


query = key = value = pe_result

mask = Variable(torch.zeros(2,4,4))
attn, p_attn = attention(query, key, value, mask=mask)

print('attn:', attn)
print(attn.shape)
print('p_attn:', p_attn)
print(p_attn.shape)

