import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class Dataset(Dataset):

    def __init__(self, para, x):
        self.data = np.load(para)[:1000]
        self.x = x

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X_train = self.data[idx,:1000]

        if self.x == 0:
            Y_train = self.data[idx, 1004]
        else:
            Y_train = self.data[idx, 1005]
        Y_train = np.array([Y_train])

        # 转为torch格式
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_train = X_train.reshape(1, 1000)
        Y_train = Y_train.reshape(1, 1)
        X_train = X_train.type(torch.FloatTensor)
        Y_train = Y_train.type(torch.FloatTensor)

        return X_train, Y_train