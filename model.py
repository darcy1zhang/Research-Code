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


# mlp relu
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#             nn.Linear(2, 4),
#             nn.ReLU(),
#             nn.BatchNorm1d(1),
#             nn.Linear(4, 4),
#             nn.ReLU(),
#             nn.BatchNorm1d(1),
#             nn.Linear(4, 4),
#             nn.ReLU(),
#             nn.BatchNorm1d(1),
#             nn.Linear(4, 4),
#             nn.Linear(4, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            # nn.Linear(8, 8),
            # nn.ReLU(),
            nn.Linear(8, 1),
        )
    def forward(self, x):
        x = self.seq(x)
        return x

# #
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#             nn.Linear(2, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x
#
# # mlp
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#             nn.Linear(2, 4),
#             nn.Linear(4, 4),
#             nn.Linear(4, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x