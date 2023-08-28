import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model import *
import matplotlib.pyplot as plt


# 生成测试用时间序列数据
x = np.load("../data/simu_10000_0.1_141_178_test.npy")[528,:1000]
tmp = np.load("../data/simu_10000_0.1_141_178_test.npy")[528,:1000]
x = torch.from_numpy(x)
x = x.type(torch.FloatTensor)
x = x.reshape(1,-1)
x = np.repeat(x[np.newaxis,:], 16, axis=0)
# print(x.shape)

x.requires_grad_(True)
print(x.requires_grad)

model = VGG()
model = torch.load("../pth/D_cnn.pth" ,map_location = torch.device('cpu'))
model.train()
model.zero_grad()


output = model(x)
target = torch.ones_like(output)
target = target * 90


criterion = nn.L1Loss()
# criterion = criterion.cuda()
loss = criterion(output, target)
loss = loss.mean()

# for param in model.parameters():
#     print(param.grad)

loss.backward()

# for param in model.parameters():
#     print(param.grad)



grad = x.grad   # (batch_size, seq_len)
grad_mean = grad.mean(dim=0)

# grad_mean = grad_mean.squeeze()
# grad_mean = grad_mean.unsqueeze(0)
grad_mean = grad_mean.flatten() * tmp
grad_mean = grad_mean / grad_mean.max()
tmp = tmp / tmp.max()

# 绘制热力图
# plt.imshow(grad_mean.numpy(),
#            cmap='hot', aspect='auto', interpolation='nearest')

# 绘制曲线
plt.plot(tmp)
plt.plot(grad_mean, 'o')
#
# # 设置标题等
# plt.title("Gradient Visualization")
# plt.xlabel("Time Step")
# plt.ylabel("Gradient Magnitude")
# plt.colorbar()
plt.show()