import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 8, 5),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.Conv1d(8, 8, 5),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 16, 5),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv1d(16, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )


        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, 5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.AvgPool1d(kernel_size=3, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(32 * 48, 10)
        )

        self.out = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(16, -1, 32 * 48)  # flatten the tensor

        x = self.fc1(x)
        output = self.out(x)

        return output


if __name__ == "__main__":
    model = VGG()
    input = torch.ones((16, 1, 1000))
    writer = SummaryWriter("./log")
    writer.add_graph(model, input)
    writer.close()