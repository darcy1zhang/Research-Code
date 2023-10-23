import torch.optim as optim
from dataset import *
from model import *
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
import tsfel

# X_train = tsfel.time_series_features_extractor(cfg_file, tmp, fs=100, window_size=250)


criterion = nn.L1Loss()
criterion = criterion.cuda()

model = MLP().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# writer = SummaryWriter('logs/fs5_sigmoid')


# tmp = np.load("./data/simu_20000_0.1_90_140_train.npy")
# max = np.max(tmp[:, 1004])
# min = np.min(tmp[:, 1004])
# print(max, min)

train_dataset = Dataset("../data/train_all_features_norm.npy", 0, "../data/train_without_resp.npy")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,drop_last=True)
test_dataset = Dataset("../data/test_all_features_norm.npy", 0, "../data/test_without_resp.npy")
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True,drop_last=True)

for epoch in range(300):
    model.train()

    loss_total = 0
    step = 0

    wzy1 = 0
    wzy2 = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        # print(target)
        # target = (target - min) / (max - min)

        pred,features = model(data)

        loss_ERM = criterion(pred, target)
        wzy1 += loss_ERM.item()


        # for sp
        sp_diff = []
        # dp_diff = []
        feature_dis = []
        # sp_contrastive_loss = np.float(0)
        sp_contrastive_loss = 0.0
        # sp_contrastive_loss = torch.from_numpy(sp_contrastive_loss)
        sp_contrastive_loss = torch.tensor(sp_contrastive_loss, requires_grad=True)

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


            sp_contrastive_loss = sp_contrastive_loss + temp_sp_loss

            sp_diff = sp_diff[:i + 1]
            # dp_diff = dp_diff[:i+ 1]
            feature_dis = feature_dis[:i + 1]

        sp_contrastive_loss = sp_contrastive_loss / (len(target[:, 0, 0]) - 1)

        wzy2 += sp_contrastive_loss.item()


        # loss = loss_ERM + sp_contrastive_loss * 100
        loss = sp_contrastive_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total = loss_total + loss.item()
        step = step + 1

    print("ERM")
    print(wzy1/step)
    print("contrast")
    print(wzy2/step)

    tmp = './pth/S_model_%d_%.4f.pth' % (epoch, loss_total/step)
    # if epoch % 10 == 0:
    # torch.save(model, tmp)
    print("S--epoch:" + str(epoch) + "    MAE:" + str(loss_total/step))

    loss_test = 0
    step = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output, _ = model(data)

            # inv_norm
            # output = output * (max - min) + min

            loss = criterion(output, target)

            loss_test = loss_test + loss.item()
            step = step + 1

        loss_mean = loss_test / step
        # if epoch % 10 == 0:
        print("epoch:" + str(epoch) + "    MAE_test:" + str(loss_mean))



#     writer.add_scalar('Training Loss', loss_total/step, epoch)
#     writer.add_scalar('Validation Loss', loss_mean, epoch)
#
# writer.close()