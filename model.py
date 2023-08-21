import torch
import torch.nn as nn
import torch.nn.init as init

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#             nn.Linear(5, 5),
#             nn.Sigmoid(),
#             nn.Linear(5, 5),
#             nn.Sigmoid(),
#             nn.Linear(5, 5),
#             nn.Sigmoid(),
#             nn.Linear(5, 5),
#             nn.Linear(5, 3),
#             nn.Linear(3, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x
#
#     def weights_init(m):
#         if isinstance(m, nn.Linear):
#             init.xavier_uniform_(m.weight)
#             init.zeros_(m.bias)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.seq = nn.Sequential(
        nn.Linear(1, 1),
        )
    def forward(self, x):
        x = self.seq(x)
        return x