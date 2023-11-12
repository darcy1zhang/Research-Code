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
    def __init__(self, feature_number):
        super(MLP, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(feature_number, 2 * feature_number),
            nn.ReLU(),
            nn.Linear(2 * feature_number, 2 * feature_number),
            nn.ReLU(),
            # nn.Linear(2*feature_number, 2*feature_number),
            # nn.ReLU(),
            nn.Linear(2 * feature_number, 1),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


# class VGG(nn.Module):
#     def __init__(self):
#         super(VGG, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(1, 8, 51, padding=25),
#             nn.BatchNorm1d(8),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(8, 8, 51, padding=25),
#             nn.BatchNorm1d(8),
#             nn.LeakyReLU(),
#
#             nn.AvgPool1d(kernel_size=3, stride=2)
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(8, 16, 51, padding=25),
#             nn.BatchNorm1d(16),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(16, 32, 51, padding=25),
#             nn.BatchNorm1d(32),
#             nn.LeakyReLU(),
#
#             nn.AvgPool1d(kernel_size=3, stride=2)
#         )
#
#         self.conv3 = nn.Sequential(
#             nn.Conv1d(32, 64, 51, padding=25),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(64, 64, 51, padding=25),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(64, 64, 51, padding=25),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(64, 64, 51, padding=25),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(),
#
#             nn.AvgPool1d(kernel_size=3, stride=2)
#         )
#
#         self.conv4 = nn.Sequential(
#             nn.Conv1d(64, 32, 5, padding=2),
#             nn.BatchNorm1d(32),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(32, 32, 5, padding=2),
#             nn.BatchNorm1d(32),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(32, 32, 5, padding=2),
#             nn.BatchNorm1d(32),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(32, 32, 5, padding=2),
#             nn.BatchNorm1d(32),
#             nn.LeakyReLU(),
#
#             nn.AvgPool1d(kernel_size=3, stride=2)
#         )
#
#         self.fc1 = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(32 * 61, 10)
#         )
#
#         self.out = nn.Linear(10, 1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#
#         x = x.view(16, -1, 32 * 61)  # flatten the tensor
#
#         x = self.fc1(x)
#         output = self.out(x)
#
#         return output


# class LR(nn.Module):
#     def __init__(self):
#         super(LR, self).__init__()
#         self.fc = nn.Linear(7, 1)
#
#     def forward(self, x):
#         x = self.fc(x)
#         return x