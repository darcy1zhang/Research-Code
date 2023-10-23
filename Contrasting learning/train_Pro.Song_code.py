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

l2_lambda = hparams.l2_lambda
loss_best = hparams.loss_best
lamda_contrast = 0.1





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

    for batch_idx, (data, target) in enumerate(train_loader):
        # print("%d"%(batch_idx))
        data, target = data.cuda(), target.cuda()
        features, output = model(data)
        loss_ERM = criterion(output, target)
        wzy1 += loss_ERM.item()

        # L2正则化
        # l2_reg = torch.tensor(0.).to("cuda:0")
        # for param in model.parameters():
        #     l2_reg += torch.norm(param, 2)
        # loss = loss + l2_lambda * l2_reg

        # for sp
        sp_diff = []
        # dp_diff = []
        feature_dis = []
        # sp_contrastive_loss = np.float(0)
        sp_contrastive_loss = 0.0
        # sp_contrastive_loss = torch.from_numpy(sp_contrastive_loss)
        sp_contrastive_loss = torch.tensor(sp_contrastive_loss, requires_grad=True)

        # sp_contrastive_loss.requires_grad = True
        # dp_contrastive_loss = 0
        for i in range(len(target[:, 0, 0]) - 1):
            for j in range(i + 1, len(target[:, 0, 0])):
                diff = torch.abs(target[i, 0, 0] - target[j, 0, 0])
                # diff2 = torch.abs(labels[i,1] - labels[j,1])
                sp_diff.append(diff)
                # dp_diff.append(diff2)
                dis = torch.sum(features[i, :, :] * features[j, :, :]) / (
                            torch.norm(features[i, :, :]) * torch.norm(features[j, :, :]))
                feature_dis.append(torch.exp(dis))
            sorted_id_diff = sorted(range(len(sp_diff)), key=lambda k: sp_diff[k])
            # sorted_id_diff2 = sorted(range(len(dp_diff)), key=lambda k: dp_diff[k])
            # sorted_feature_dis = torch.tensor(feature_dis)[sorted_id_diff]
            # sorted_feature_dis2 = torch.tensor(feature_dis)[sorted_id_diff2]

            sorted_feature_dis = []
            for k in range(len(sorted_id_diff)):
                sorted_feature_dis.append(feature_dis[sorted_id_diff[k]])

            temp_sp_loss = torch.tensor(0.0, requires_grad=True)
            # temp_dp_loss = 0
            for p in range(len(sorted_feature_dis) - 1):
                sum = torch.tensor(0.0, requires_grad=True)
                for q in range(p + 1, len(sorted_feature_dis)):
                    sum = sum + sorted_feature_dis[q]
                # temp_sp_loss = temp_sp_loss + torch.log(sorted_feature_dis[j]) - torch.log(torch.sum(sorted_feature_dis[j+1:]))
                temp_sp_loss = temp_sp_loss + torch.log(sorted_feature_dis[p]) - torch.log(sum)
                # temp_dp_loss += torch.log(sorted_feature_dis2[j]) - torch.log(torch.sum(sorted_feature_dis2[j+1:]))
            temp_sp_loss = temp_sp_loss / (len(sorted_feature_dis) - 1)
            # temp_dp_loss /= (len(sorted_feature_dis2) - 1)

            # 调整contrastive loss的比例
            sp_contrastive_loss = sp_contrastive_loss + temp_sp_loss

            sp_diff = sp_diff[:i + 1]
            # dp_diff = dp_diff[:i+ 1]
            feature_dis = feature_dis[:i + 1]

        sp_contrastive_loss = sp_contrastive_loss / (len(target[:, 0, 0]) - 1)
        # dp_contrastive_loss/= (len(labels[:,0]) - 1)

        # print(loss)
        # print(sp_contrastive_loss)
        wzy2 += sp_contrastive_loss.item()

        loss = loss_ERM + sp_contrastive_loss * 10

        optimizer.zero_grad()
        loss.backward()
        # sp_contrastive_loss.backward()
        optimizer.step()

        loss_total = loss_total + loss.item()
        step = step + 1

    print("ERM")
    print(wzy1 / step)
    print("contrastive")
    print(wzy2 / step)

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
