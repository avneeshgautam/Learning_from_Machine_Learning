import torch
import torch.nn as nn
import math
import numpy as np

# class InputEmbedding(nn.modules):
#     def __init__(self,d_model:int, vocab_size:int):
#         super().__init__()
#         self.d_model = d_model
#         self.vocab_size = vocab_size
#         self.embedding = nn.Embedding(vocab_size, d_model)

#     def forward(self, x):
#         return self.embedding(x) * math.sqrt(self.d_model)


def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T

def scaled_dot_product_attention(q,k,v,mask=None):
    d_k = q.shape[-1]
    scaled = np.matmul(q,k.T) / math.sqrt(d_k)

    if mask is not None:
        scaled += mask
    
    attention = softmax(scaled)
    out = np.matmul(attention, v)
    return out,attention




L,d_k, d_v = 4,8,8   # L = length of input sequence (my name is ajay) , d_k = dimension of key, d_v = dimension of value
q = np.random.rand(L,d_k)  # query
k = np.random.rand(L,d_k)  # key
v = np.random.rand(L,d_v)  # value
mask = np.tril(np.ones((L,L)))
mask[mask == 0] = -np.inf
mask[mask == 1] = 0
new_values, attention = scaled_dot_product_attention(q,k,v,mask)
print(f"new_values: {new_values}, attention: {attention}")