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

class MultiheadCrossAttention(nn.Module):
    def __init__(self, d_model, nums_head):
        super().__init__()
        self.d_model = d_model
        self.nums_head = nums_head
        self.head_dim = d_model // nums_head
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def scaled_dot_product(self,q, k, v, mask = None):
       
        d_k = q.size()[-1]
        scaled = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)  
        if mask is not None:
            print(f"-- ADDING MASK of shape {mask.size()} --") 
            scaled +=mask
        
        attention = F.softmax(scaled, dim = -1) 
        values = torch.matmul(attention, v)
        return values, attention


    def forward(self, x, y,  mask=None):
        batch_size, sequence_length, d_model = x.size()
        print("x.size() : ",x.size())
        kv = self.kv_layer(x)
        print(f"kv.size(): {kv.size()}")
        q = self.q_layer(y)
        print(f"q.size(): {q.size()}")

        kv = kv.reshape(batch_size, sequence_length, self.nums_head, 2 * self.head_dim)
        print(f"kv.size(): {kv.size()}")
        q = q.reshape(batch_size, sequence_length, self.nums_head, self.head_dim)
        print(f"q.size(): {q.size()}")
        kv = kv.permute(0,2,1,3)
        print(f"kv.size(): {kv.size()}")
        q = q.permute(0,2,1,3)
        print(f"q.size(): {q.size()}")
        k, v = kv.chunk(2, dim = -1)
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



class DecoderLayer(nn.Module):
    def __init__(self,d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model , num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.encoder_decoder_attention = MultiheadCrossAttention(d_model, num_heads)
        self.norm2 = LayerNormalization([d_model])
        self.dropout2 = nn.Dropout(p = drop_prob)
        self.ffn = PositionwiseFeedForward(d_model = d_model, hidden = ffn_hidden, drop_prob = drop_prob)
        self.norm3 = LayerNormalization([d_model])
        self.dropout3 = nn.Dropout(p = drop_prob)

    
    def forward(self, x, y, decoder_mask):
        _y = y
        print("Masked self Attention ")
        y = self.self_attention(y, mask = decoder_mask)
        print("DropOUT 1 ")
        y  = self.dropout1(y)
        print("ADD + NORM ")
        y = self.norm1(y + _y)

        _y = y
        print("Cross Attention")
        y = self.encoder_decoder_attention(x,y,mask = None)
        print("DropOUT 2 ")
        y = self.dropout2(y)
        print("Add+ Norm 2")
        y = self.norm2(y + _y)

        _y = y
        print("Feed forwards")
        y = self.ffn(y)
        print("DropOUT 3 ")
        y = self.dropout2(y)
        print("Add+ Norm 3")
        y = self.norm2(y + _y)

        return y



class SequentialDecoer(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        for modules in self._modules.values():
            x = modules(x,y,mask)
        
        return y


class Decoder(nn.Module):
    def __init__(self,d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = SequentialDecoer(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob, num_layers) for _ in range(num_layers)])

    def forward(self,x,y,mask):
        # x : 30 x 200 x 512
        # y = 20 x 200 x 512
        y = self.layers(x,y,mask) 
        return y

        
d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 2048
num_layers = 5



x = torch.randn((batch_size , max_sequence_length, d_model)) # english sentence positional encoded and also output of encoder
y = torch.randn((batch_size , max_sequence_length, d_model)) # hindi sentence positional encoded and also output of encoder

mask = torch.full([max_sequence_length , max_sequence_length] , float('-inf'))
mask = torch.triu(mask, diagonal = 1)
decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
out = decoder(x,y,mask)
