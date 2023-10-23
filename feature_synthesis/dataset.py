import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from get_features import *
from sklearn.preprocessing import MinMaxScaler

class Dataset(Dataset):

    def __init__(self, data, x, target):
        self.data = np.load(data)
        self.x = x
        self.target = np.load(target)


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X_train = self.data[idx,:]

        if self.x == 0:
            Y = self.target[idx, 1004]
        else:
            Y = self.target[idx, 1005]

        Y_train = np.array([Y])

        # 转为torch格式
        X_train = np.array([X_train])
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_train = X_train.reshape(1, 77)
        Y_train = Y_train.reshape(1, 1)
        X_train = X_train.type(torch.FloatTensor)
        Y_train = Y_train.type(torch.FloatTensor)

        return X_train, Y_train