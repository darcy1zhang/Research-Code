import torch
import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Linear(77, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.seq2 = nn.Sequential(
            nn.Linear(8, 1),
        )
    def forward(self, x):
        feature = self.seq1(x)
        pred = self.seq2(feature)

        return pred, feature

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.seq = nn.Sequential(
#         nn.Linear(1, 1),
#         )
#     def forward(self, x):
#         x = self.seq(x)
#         return x