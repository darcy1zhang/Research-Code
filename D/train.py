import torch.optim as optim
from dataset import *
from model import *
from hparams import *

# writer = SummaryWriter("logs")

hparams = hparams()
model = VGG().cuda()
criterion = nn.L1Loss()
criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=hparams.lr)
#
l2_lambda = hparams.l2_lambda
loss_best = hparams.loss_best
#
# train_dataset = Dataset("./data/simu_20000_0.1_90_140_train.npy", 0)
# train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
# test_dataset = Dataset("./data/simu_10000_0.1_141_178_test.npy", 0)
# test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=True)
#
# for epoch in range(300):
#     model.train()
#
#     loss_total = 0
#     step = 0
#
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.cuda(), target.cuda()
#         output = model(data)
#         loss = criterion(output, target)
#
#         # L2正则化
#         l2_reg = torch.tensor(0.).to("cuda:0")
#         for param in model.parameters():
#             l2_reg += torch.norm(param, 2)
#         loss += l2_lambda * l2_reg
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         loss_total = loss_total + loss.item()
#         step = step + 1
#
#     tmp = './pth/S_model_%d_%.4f.pth' % (epoch, loss_total/step)
#     if epoch % 30 == 0:
#         torch.save(model, tmp)
#     print("S--epoch:" + str(epoch) + "    MAE:" + str(loss_total/step))
#
#     loss_test = 0
#     step = 0
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(test_loader):
#             data, target = data.cuda(), target.cuda()
#             output = model(data)
#             loss = criterion(output, target)
#
#             loss_test = loss_test + loss.item()
#             step = step + 1
#
#         loss_mean = loss_test / step
#         print("epoch:" + str(epoch) + "    MAE_test:" + str(loss_mean))
#     writer.add_scalars("S_loss", {"train": loss_total / step, "validation": loss_mean}, epoch)

train_dataset = Dataset("../data/BPD_S_90_110.npy", 0)
train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, drop_last=True)
test_dataset = Dataset("../data/BPD_S_110_130.npy", 0)
test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=True, drop_last=True)

for epoch in range(300):
    model.train()

    loss_total = 0
    step = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)

        # L2正则化
        l2_reg = torch.tensor(0.).to("cuda:0")
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)
        loss += l2_lambda * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total = loss_total + loss.item()
        step = step + 1

    tmp = './pth/D_model_%d_%.4f.pth' % (epoch, loss_total/step)
    if epoch % 6 == 0:
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
    # writer.add_scalars("D_loss", {"train": loss_total / step, "validation": loss_mean}, epoch)
