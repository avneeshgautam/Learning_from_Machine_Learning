import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# class LayerNormalization2(nn.Module):
#     def __init__(self, features: int, eps:float=10**-6) -> None:
#         super().__init__()
#         self.eps = eps
#         self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
#         self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

#     def forward(self, x):
#         # x: (batch, seq_len, hidden_size)
#          # Keep the dimension for broadcasting
#         mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
#         # Keep the dimension for broadcasting
#         std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
#         y = (x - mean) / (std + self.eps ) # (batch, seq_len, hidden_size)
#         # eps is to prevent dividing by zero or when std is very small
#         out = self.alpha * y + self.bias
#         return out

def get_device():
    return torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

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
        # print(f"Mean ({mean.size()})")
        var = ((inputs-mean)**2).mean(dim = dims,keepdim = True)
        std = (var + self.eps).sqrt()
        # print(f"Standard Deviation  ({std.size()})")
        y = (inputs - mean) / std
        # print(f"y: {y.size()}")
        out = self.gamma * y + self.beta
        # print(f"self.gamma: {self.gamma.size()}, self.beta: {self.beta.size()}")
        # print(f"out: {out.size()}")
        return out

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



class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token=True, end_token=True):

        def tokenize(sentence, start_token=True, end_token=True):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, x, start_token, end_token): # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # self.input_dim = input_dim      # 512
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
        
        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        q = q.reshape(batch_size,sequence_length, self.num_heads, self.head_dim)    # 1, 4, 8 , 64
        k = k.reshape(batch_size,sequence_length, self.num_heads, self.head_dim)    # 1, 4, 8 , 64
        v = v.reshape(batch_size,sequence_length, self.num_heads, self.head_dim)    # 1, 4, 8 , 64
        q = q.permute(0,2,1,3)  # 1, 4, 8 , 64 -> 1, 8, 4, 64
        k = k.permute(0,2,1,3)  # 1, 4, 8 , 64 -> 1, 8, 4, 64
        v = v.permute(0,2,1,3)  # 1, 4, 8 , 64 -> 1, 8, 4, 64
        # print(q.size(), k.size(), v.size())
        attention, values = self.scaled_dot_product(q,k,v,mask)  # attention = [1, 8, 4, 4] , values = [1, 8, 4, 64]
        values = values.permute(0,2,1,3)  # 1, 8, 4, 64 -> 1, 4, 8, 64 

        # h is concatenated values of all heads
        h = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim) # 1, 4, 8, 64 -> 1, 4, 512
        out = self.w_o(h)  # 1, 4, 512
        return out

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # self.input_dim = input_dim      # 512
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

    def forward(self, x, y, mask=None):
        batch_size, sequence_length, input_dim = x.size()                # [4, 512 ] -> # [1, 4, 512]  ## batch for parralizartroin
        q = self.wq(y)   # 1, 4, 512   ## input from encoder
        k = self.wk(x)   # 1, 4, 512   ## decoder input
        v = self.wv(x)   # 1, 4, 512      

        q = q.reshape(batch_size,sequence_length, self.num_heads, self.head_dim)    # 1, 4, 8 , 64
        k = k.reshape(batch_size,sequence_length, self.num_heads, self.head_dim)    # 1, 4, 8 , 64
        v = v.reshape(batch_size,sequence_length, self.num_heads, self.head_dim)    # 1, 4, 8 , 64
        q = q.permute(0,2,1,3)  # 1, 4, 8 , 64 -> 1, 8, 4, 64
        k = k.permute(0,2,1,3)  # 1, 4, 8 , 64 -> 1, 8, 4, 64
        v = v.permute(0,2,1,3)  # 1, 4, 8 , 64 -> 1, 8, 4, 64

        attention, values = self.scaled_dot_product(q,k,v,mask)  # attention = [1, 8, 4, 4] , values = [1, 8, 4, 64]
        values = values.permute(0,2,1,3) # 1, 8, 4, 64 -> 1, 4, 8, 64 
        # h is concatenated values of all heads
        h = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim) # 1, 4, 8, 64 -> 1, 4, 512
        out = self.w_o(h)  # 1, 4, 512
        return out
    

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, ffn_hidden, drop_prob=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self,d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer,self).__init__()
        self.attention = MultiHeadAttention(d_model = d_model, num_heads = num_heads)
        self.norm1 = LayerNormalization(parameters_shape = [d_model])
        self.dropout1 = nn.Dropout(drop_prob)
        self.fnn = PositionwiseFeedforward(d_model = d_model, ffn_hidden = ffn_hidden,drop_prob = drop_prob)
        self.norm2 = LayerNormalization(parameters_shape = [d_model])
        self.dropout2 = nn.Dropout(drop_prob)
    def forward(self, x, self_attention_mask):
        residual_x = x.clone() # [batch, 4, 512]
        x = self.attention(x, mask = None) # mask = None
        x = self.dropout1(x)    
        x = self.norm1(x + residual_x)
        residual_x = x.clone() # [batc, 4, 512]
        x = self.fnn(x) 
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x
    
class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(parameters_shape = [d_model])
        self.dropout1 = nn.Dropout(drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.norm2 = LayerNormalization(parameters_shape = [d_model])
        self.dropout2 = nn.Dropout(drop_prob)

        self.fnn = PositionwiseFeedforward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNormalization(parameters_shape = [d_model])
        self.dropout3 = nn.Dropout(drop_prob)
    
    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y.clone()
        y = self.self_attention(y, mask = self_attention_mask)
        y = self.dropout1(y)
        y = self.norm1(y + _y)

        _y = y.clone()
        y = self.encoder_decoder_attention(x, y , mask = cross_attention_mask)
        y = self.dropout2(y)
        y = self.norm2(y + _y)

        _y = y.clone() 
        y = self.fnn(y)
        y = self.dropout3(y)
        y = self.norm3(y + _y)

        return y

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask  = inputs
        for module in self._modules.values():  # calling every single layer
            y = module(x, y, self_attention_mask, cross_attention_mask) # getting new value of y(hindi) from each layer
        return y


class Decoder(nn.Module):
    def __init__(self,
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()         
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)          
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) 
                                          for _ in range(num_layers)])
        # original nn.Sequential will not pass more than one parameter thatswhy we are using own sequential
    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        # x = english sentence batch_size x seq_length x 512
        # y hindi sentence batch_size x seq_length x 512
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y


class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_sequence_length, 
                hn_vocab_size,
                english_to_index,
                hindi_to_index,
                START_TOKEN, 
                END_TOKEN, 
                PADDING_TOKEN
                ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, 
                               english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length,
                                hindi_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, hn_vocab_size)
        self.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, 
                x, # batch of english sentences
                y, # batch of hindi sentences
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make this true
                dec_end_token=False): # x, y are batch of sentences
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, 
                           start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out

# d_model = 512
# num_heads  = 8
# drop_prob = 0.1
# batch_size = 30
# max_sequence_length = 200
# ffn_hidden = 2048 # increase size to 2048 layer to learn better
# num_layers = 5


# # encoder = Encoder(d_model , ffn_hidden, num_heads, drop_prob, num_layers)
# # x = torch.randn(batch_size, max_sequence_length, d_model)
# # print(f"Input shape: {x.size()}")
# # output = encoder(x)
# # print(f"Output shape: {output.size()}") 

# x = torch.randn((batch_size, max_sequence_length, d_model)) # english sentence positional encoded and also output of encoder
# y = torch.randn((batch_size, max_sequence_length, d_model)) # hindi sentence positional encoded
# mask = torch.full([max_sequence_length, max_sequence_length], float('-inf'))
# mask = torch.triu(mask, diagonal = 1)
# decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
# out = decoder(x,y,mask)
# print(f"Output shape: {out.size()}")

