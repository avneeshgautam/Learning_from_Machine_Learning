import torch
z = torch.tensor([5, 7, 10])

softmax = torch.exp(z) / torch.sum(torch.exp(z))

print(softmax)