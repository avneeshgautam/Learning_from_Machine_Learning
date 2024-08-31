import torch
import torch.nn as nn
x = torch.Tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])


#### screach
sm = torch.sum(x, axis=1, keepdims=True)
softmax = torch.exp(x) / torch.sum(torch.exp(x), axis=1, keepdims=True)
print(softmax)


## inbuilt

softmax_layer = nn.Softmax(dim=1)
output = softmax_layer(x)
print(output)