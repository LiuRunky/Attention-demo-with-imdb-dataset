import torch
import torch.nn as nn


class NaiveMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(NaiveMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.wq = nn.Linear(in_features=embed_dim, out_features=self.embed_dim, bias=False)
        self.wk = nn.Linear(in_features=embed_dim, out_features=self.embed_dim, bias=False)
        self.wv = nn.Linear(in_features=embed_dim, out_features=self.embed_dim, bias=False)

    def forward(self, query, key, value):
        n = query.shape[1]
        q, k, v = self.wq(query), self.wk(key), self.wv(value)  # [batch_size, n, num_heads * head_dim]

        q = q.reshape([-1, self.num_heads, n, self.head_dim])  # [batch_size, num_heads, n, head_dim]
        q = q.reshape([-1, n, self.head_dim])  # [batch_size * num_heads, n, head_dim]
        k = k.reshape([-1, self.num_heads, n, self.head_dim])
        k = k.reshape([-1, n, self.head_dim])
        v = v.reshape([-1, self.num_heads, n, self.head_dim])
        v = v.reshape([-1, n, self.head_dim])

        a = torch.bmm(q, torch.transpose(k, 1, 2))  # [batch_size * num_heads, n, n]
        a = torch.div(a, self.head_dim ** 0.5)
        a = torch.softmax(a, dim=2)

        output = torch.bmm(a, v)  # [batch_size * num_heads, n, head_dim]
        output = output.reshape([-1, n, self.num_heads * self.head_dim])  # [batch_size, n, num_heads * head_dim]
        return output, None
