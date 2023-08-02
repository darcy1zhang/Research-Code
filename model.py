import torch
import torch.nn as nn

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Sigmoid(),
#         nn.Linear(5, 5),
#         nn.Linear(5, 3),
#         nn.Linear(3, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.seq = nn.Sequential(
        nn.Linear(10, 1),
        )
    def forward(self, x):
        x = self.seq(x)
        return x