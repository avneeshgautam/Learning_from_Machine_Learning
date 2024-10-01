import numpy as np
import torch
import torch.nn as nn

## cross entorpy from screach

def cross_entropy_loss(actual, predicted):
    predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
    return - (np.sum(actual* np.log(predicted)))


y = np.array([1, 0, 0])
y_pred = np.array([0.9, 0.1,0.5])
y_pred2 = np.array([0.1, 0.9, 0])

print("loss 1 ", cross_entropy_loss(y, y_pred))
print("loss 2 ", cross_entropy_loss(y, y_pred2))


## using torch
cross_entropy = nn.CrossEntropyLoss()
y = torch.tensor([0])
y_pred = torch.tensor([[0.9, 0.1,0.5]])
y_pred2 = torch.tensor([[0.1, 0.9, 0]])

print("loss 1 ", cross_entropy(y_pred, y))
print("loss 2 ", cross_entropy(y_pred2, y))

