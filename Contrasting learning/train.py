import torch.optim as optim
from dataset import *
from model import *
from hparams import *
from contrasting_func import *

# writer = SummaryWriter("logs")

hparams = hparams()
model = VGG().cuda()
criterion = nn.L1Loss()
criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=hparams.lr)
#
l2_lambda = hparams.l2_lambda
loss_best = hparams.loss_best
lamda_contrast = 0.1
NT = NTXentLoss("cpu", 16, 0.2, True)




train_dataset = Dataset("../data/simu_20000_0.1_90_140_train.npy", 0)
train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, drop_last=True)
test_dataset = Dataset("../data/simu_10000_0.1_141_178_test.npy", 0)
test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=True, drop_last=True)

for epoch in range(1000):
    model.train()

    loss_total = 0
    step = 0
    wzy1 = 0
    wzy2 = 0
    wzy3 = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(), target.cuda()
        # data, target = data.cpu().numpy(), target.cpu().numpy()

        # data.requires_grad = True
        # target.requires_grad = True
        weak_aug, strong_aug = DataTransform(data)

        weak_aug = torch.from_numpy(weak_aug)
        weak_aug = weak_aug.type(torch.FloatTensor)

        strong_aug = strong_aug.type(torch.FloatTensor)

        weak_aug = weak_aug.cuda()
        strong_aug = strong_aug.cuda()

        strong_aug.requires_grad = True
        weak_aug.requires_grad = True

        zis, _ = model(weak_aug)
        zjs, _ = model(strong_aug)

        loss_contrast = NT(zis, zjs)


        data = data.cuda()
        target = target.cuda()

        _, output = model(data)

        loss_ERM = criterion(output, target)

        loss_reg = 0
        # L2正则化
        l2_reg = torch.tensor(0.).to("cuda:0")
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)
        loss_reg += l2_lambda * l2_reg

        optimizer.zero_grad()
        loss = loss_reg + loss_ERM + loss_contrast * lamda_contrast
        # loss = loss_reg + loss_ERM
        # loss = loss_contrast
        wzy1 += loss_reg.item()
        wzy2 += loss_ERM.item()
        wzy3 += loss_contrast.item() * lamda_contrast

        loss.backward()
        optimizer.step()

        loss_total = loss_total + loss.item()
        step = step + 1

    print("reg")
    print(wzy1/step)
    print("ERM")
    print(wzy2/step)
    print("contrast")
    print(wzy3/step)

    tmp = './pth/D_model_%d_%.4f.pth' % (epoch, loss_total/step)
    if epoch % 30 == 0:
        torch.save(model, tmp)
    print("D--epoch:" + str(epoch) + "    MAE:" + str(loss_total/step))


    loss_test = 0
    step = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            _, output = model(data)
            loss = criterion(output, target)

            loss_test = loss_test + loss.item()
            step = step + 1

        loss_mean = loss_test / step
        print("epoch:" + str(epoch) + "    MAE_test:" + str(loss_mean))
    # writer.add_scalars("D_loss", {"train": loss_total / step, "validation": loss_mean}, epoch)
