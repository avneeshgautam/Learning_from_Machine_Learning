import numpy as np 
import torch
import torch.nn as nn

def softmax_fun(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

softmax = nn.Softmax()
logits = torch.tensor([2.0, 1.0, 0.1])
output = softmax(logits)
print(output)



### own
logits = np.array([2.0, 1.0, 0.1])
softmax_values = softmax_fun(logits)

print("Softmax values:", softmax_values)