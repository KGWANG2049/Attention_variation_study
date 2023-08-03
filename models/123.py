import torch
import torch.nn.functional as F

x = torch.arange(1, 21).reshape(4, 5).float()
x_softmax = F.softmax(x, dim=-1)
y_softmax = F.softmax(x, dim=0)
print(x_softmax)
print(y_softmax)


