import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nums_head):
        super().__init__()
        self.d_model = d_model
        self.nums_head = nums_head
        self.head_dim = d_model // nums_head
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def scaled_dot_product(self,q, k, v, mask = None):

        # maskMat = np.tril(np.ones((L,L)))
        # maskMat[maskMat == 0] = -np.infty
        # maskMat[maskMat == 1] = 0
       
        d_k = q.size()[-1]
        scaled = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)  
        if mask is not None:
            print(f"-- ADDING MASK of shape {mask.size()} --") 
            scaled +=mask
        
        attention = F.softmax(scaled, dim = -1) 
        values = torch.matmul(attention, v)
        return values, attention


    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size()
        print("x.size() : ",x.size())
        qkv = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(batch_size, sequence_length, self.nums_head, 3 * self.head_dim)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0,2,1,3)
        print(f"qkv.size(): {qkv.size()}")
        q, k, v = qkv.chunk(3, dim = -1)
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}, ")
        values, attention = self.scaled_dot_product(q,k,v, mask)
        print(f"values.size(): {values.size()}, attention.size:{ attention.size()} ")
        values = values.reshape(batch_size, sequence_length, self.nums_head * self.head_dim)
        print(f"values.size(): {values.size()}")
        out = self.linear_layer(values)
        print("out.size() : ",out.shape)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
    
    def forward(self):
        even_i = torch.arange(0,self.d_model,2).float()     # even positions
        odd_i = torch.arange(1,self.d_model,2).float()      # odd positions
        denominator = torch.pow(10000,(odd_i -1 )/ d_model)   
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length,1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE,odd_PE],dim = 2)   # one for seq and one for d_model
        PE = torch.flatten(stacked , start_dim=1, end_dim=2)
        print("PE.size()",PE.size())
        return PE


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps = 1e-5):
        super().__init__()
        self.parameter_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i+1) for i in range(len(self.parameter_shape))]
        mean = inputs.mean(dim = dims, keepdim = True)
        print(f"Mean ({mean.size()})")
        var = ((inputs-mean)**2).mean(dim = dims,keepdim = True)
        std = (var + self.eps).sqrt()
        print(f"Standard Deviation  ({std.size()})")
        y = (inputs - mean) / std
        print(f"y: {y.size()}")
        out = self.gamma * y + self.beta
        print(f"self.gamma: {self.gamma.size()}, self.beta: {self.beta.size()}")
        print(f"out: {out.size()}")
        return out



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)


    def forward(self, x):
        x = self.linear1(x)
        print(f"x after first linear layer: {x.size()}")
        x = self.relu(x)
        print(f"x after activation: {x.size()}")
        x = self.dropout(x)
        print(f"x after dropout: {x.size()}")
        x = self.linear2(x)
        print(f"x after Second linear layer: {x.size()}")

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, drop_prob, ffn_hidden):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model = d_model, nums_head=num_heads)
        self.norm1 = LayerNormalization(parameters_shape = [d_model])
        self.norm2 = LayerNormalization(parameters_shape = [d_model])
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.dropout2= nn.Dropout(p = drop_prob)
        self.ffn = PositionwiseFeedForward(d_model = d_model , hidden = ffn_hidden, drop_prob= drop_prob)

    
    def forward(self, x):
        residual_x = x
        print("------- ATTENTION 1 ------")
        x = self.attention(x, mask=None)
        print("------- DROPOUT 1 ------")
        x = self.dropout1(x)
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        x = self.norm1(x+residual_x)
        residual_x = x
        print("------- Feed forward network------")
        x = self.ffn(x)
        print("------- DROPOUT 2 ------")
        x = self.dropout2(x)
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        x = self.norm2(x + residual_x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, drop_prob, num_layers, ffn_hidden):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, num_heads, drop_prob,ffn_hidden)
                                     for _ in range(num_layers)])
    def forward(self,x):
        s = self.layers(x)
        return x
    

d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 2048
num_layers = 5

encoder = Encoder(d_model = d_model, num_heads = num_heads,drop_prob= drop_prob, num_layers=num_layers,ffn_hidden=ffn_hidden)
x = torch.randn( (batch_size, max_sequence_length, d_model) ) # includes positional encoding
out = encoder(x)
