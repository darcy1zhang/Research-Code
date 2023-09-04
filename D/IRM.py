import torch.optim as optim
import torch.nn as nn
import torch

from dataset import *
from model import *
from hparams import *

# def compute_penalty(model, data, target, loss_fn):
#     # 计算IRM惩罚项
#     output = model(data)
#     loss = loss_fn(output, target)
#     gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
#     penalty = sum(torch.sum(grad ** 2) for grad in gradients)
#     return penalty

hparams = hparams()
model = VGG().cuda()
criterion = nn.L1Loss()
criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=hparams.lr)

l2_lambda = hparams.l2_lambda
loss_best = hparams.loss_best
irm_lambda = 0.05  # 调整IRM的lambda值

env1_train_dataset = Dataset("../data/S_90_110.npy", 0)
env1_train_loader = DataLoader(env1_train_dataset, batch_size=hparams.batch_size, shuffle=True, drop_last=True)
env2_train_dataset = Dataset("../data/S_130_150.npy", 0)
env2_train_loader = DataLoader(env2_train_dataset, batch_size=hparams.batch_size, shuffle=True, drop_last=True)
test_dataset = Dataset("../data/S_160_180.npy", 0)
test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=True,drop_last=True)

def compute_kl_divergence(p, q):
    return (p * torch.log(p / q)).sum()

for epoch in range(300):
    model.train()

    loss_total = 0
    step = 0

    # for batch_idx, (data1, target1, data2, target2) in enumerate(
    #         zip(env1_train_loader, env1_target_loader, env2_train_loader, env2_target_loader)):
    #     data1, target1 = data1.cuda(), target1.cuda()
    #     data2, target2 = data2.cuda(), target2.cuda()

    for batch_idx, ((data1, target1), (data2, target2)) in enumerate(zip(env1_train_loader,env2_train_loader)):
        data1, target1 = data1.cuda(), target1.cuda()
        data2, target2 = data2.cuda(), target2.cuda()
        output1 = model(data1)
        loss1 = criterion(output1, target1)
        output2 = model(data2)
        loss2 = criterion(output2, target2)

        # # L2正则化
        # l2_reg = torch.tensor(0.).to("cuda:0")
        # for param in model.parameters():
        #     l2_reg += torch.norm(param, 2)
        # loss += l2_lambda * l2_reg

        p = torch.softmax(output1, dim=1)  # 将输出转化为概率分布
        q = torch.softmax(output2 - 40, dim=1)
        kl_divergence = compute_kl_divergence(p, q)

        # 计算IRM惩罚项并添加到损失中
        loss = loss1 + loss2 + irm_lambda * kl_divergence

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total = loss_total + loss.item()
        step = step + 1

    tmp = './pth/S_model_%d_%.4f.pth' % (epoch, loss_total/step)
    if epoch % 30 == 0:
        torch.save(model, tmp)
    print("D--epoch:" + str(epoch) + "    MAE:" + str(loss_total/step))

    loss_test = 0
    step = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)

            loss_test = loss_test + loss.item()
            step = step + 1

        loss_mean = loss_test / step
        print("epoch:" + str(epoch) + "    MAE_test:" + str(loss_mean))
