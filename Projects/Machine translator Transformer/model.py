import torch
import torch.nn as nn
import torch.nn.functional as F
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

    def forward(self):
        even_i = torch.arange(0,self.d_model,2).float()    # even positions 
        denominator = torch.pow(10000,(even_i / self.d_model)) # 10000^(2i/d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length,1) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 1]
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        PE = torch.zeros(self.max_sequence_length, self.d_model) # [10, 512]
        PE[:,0::2] = even_PE
        PE[:,1::2] = odd_PE
        return PE 

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim      # 512
        self.d_model = d_model          # 512
        self.num_heads = num_heads      # 8
        self.head_dim = d_model // num_heads # 512 / 8 = 64
        self.wq = nn.Linear(d_model, d_model)   # 512, 512
        self.wk = nn.Linear(d_model, d_model)   # 512, 512
        self.wv = nn.Linear(d_model, d_model)   # 512, 512
        self.w_o = nn.Linear(d_model, d_model)  # 512, 512

    def scaled_dot_product(self,q,k,v, mask = None):
        d_k = q.size()[-1]
        scaled = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k) # [1, 8, 4, 64] * [1, 8, 64, 8] -> [1, 8, 4, 4]
        
        if mask is not None:
            scaled += mask # [1, 8, 4, 4] + [1, 8, 4, 4] -> [1, 8, 4, 4]
        
        attention = F.softmax(scaled, dim=-1)  # apply only on last dimention # [1, 8, 4, 4]
        values = torch.matmul(attention, v) # [1, 8, 4, 4] * [1, 8, 4, 64] -> [1, 8, 4, 64]

        return attention, values # [1, 8, 4, 4], [1, 8, 4, 64]

    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size() # [4, 512 ] -> # [1, 4, 512]  ## batch for parralizartroin
        q = self.wq(x)   # 1, 4, 512
        k = self.wk(x)   # 1, 4, 512
        v = self.wv(x)   # 1, 4, 512

        q = q.reshape(batch_size,sequence_length, self.num_heads, self.head_dim)    # 1, 4, 8 , 64
        k = k.reshape(batch_size,sequence_length, self.num_heads, self.head_dim)    # 1, 4, 8 , 64
        v = v.reshape(batch_size,sequence_length, self.num_heads, self.head_dim)    # 1, 4, 8 , 64
        q = q.permute(0,2,1,3)  # 1, 4, 8 , 64 -> 1, 8, 4, 64
        k = k.permute(0,2,1,3)  # 1, 4, 8 , 64 -> 1, 8, 4, 64
        v = v.permute(0,2,1,3)  # 1, 4, 8 , 64 -> 1, 8, 4, 64

        attention, values = self.scaled_dot_product(q,k,v,mask)  # attention = [1, 8, 4, 4] , values = [1, 8, 4, 64]
        values = values.permute(0,2,1,3)  # 1, 8, 4, 64 -> 1, 4, 8, 64 
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim) # 1, 4, 8, 64 -> 1, 4, 512

        out = self.w_o(values)  # 1, 4, 512
        return out


# L,d_k, d_v = 4,8,8   # L = length of input sequence (my name is ajay) , d_k = dimension of key, d_v = dimension of value
# q = np.random.rand(L,d_k)  # query
# k = np.random.rand(L,d_k)  # key
# v = np.random.rand(L,d_v)  # value

input_dim = 512
d_model = 512
num_heads = 8

batch_size = 1
sequence_length = 4
x = torch.rand(batch_size, sequence_length, input_dim)

model = MultiHeadAttention(input_dim, d_model, num_heads)
out = model(x)
