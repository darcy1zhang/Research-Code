from model import *
from utils import *
from dataset import *

def prediction(data_file):
    model = torch.load("./pth/S_model_120_9.6622.pth" ,map_location = torch.device('cpu'))
    dataset_test = Dataset(data_file, 0)
    train_loader = DataLoader(dataset_test, batch_size=16, shuffle=False, drop_last=True)

    criterion = nn.L1Loss()


    loss_total = 0
    batch_num = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(), target.cuda()
        output = model(data)

        if 'dp_prediction' in locals():
            dp_prediction = torch.cat((dp_prediction, output), dim=0)
        else:
            dp_prediction = output

        loss = criterion(output, target)
        # print(loss.mean().item())
        loss_total = loss_total + loss.item()
        batch_num = batch_num + 1

    dp_prediction = dp_prediction.reshape(-1)
    dp_MAE = loss_total / batch_num

    # 预测S
    # model = torch.load("./pth/S_cnn2.pth", map_location=torch.device('cpu'))
    # dataset_test = Dataset(data_file, 0)
    # train_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)
    #
    # loss_total = 0
    # batch_num = 0
    #
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     # data, target = data.cuda(), target.cuda()
    #     output = model(data)
    #
    #     if 'sp_prediction' in locals():
    #         sp_prediction = torch.cat((sp_prediction, output), dim=0)
    #     else:
    #         sp_prediction = output
    #
    #     loss = criterion(output, target)
    #     # print(loss.mean().item())
    #     loss_total = loss_total + loss.item()
    #     batch_num = batch_num + 1
    #
    # sp_prediction = sp_prediction.reshape(-1)
    # sp_MAE = loss_total / batch_num

    # return dp_MAE, dp_prediction, sp_MAE, sp_prediction
    return dp_MAE, dp_prediction

if __name__ == "__main__":
    data_file = "../data/S_160_180.npy"
    prediction_file = '../data/prediction.npy'

    # dp_MAE, dp_prediction, sp_MAE, sp_prediction = prediction(data_file)
    dp_MAE, dp_prediction = prediction(data_file)
    dp_prediction = dp_prediction.detach().numpy()
    # sp_prediction = sp_prediction.detach().numpy()
    np.save(prediction_file, dp_prediction)

    # print('===============')
    # print('Systolic MAE:', sp_MAE)
    print('===============')
    print('MAE:', dp_MAE)
    print('===============')

    data_set = np.load(data_file)
    # sp_label = data_set[:, -2]
    dp_label = data_set[:992, -2]

    dp_prediction = np.load(prediction_file)

    # plot_2vectors(sp_label, sp_prediction, 'sp')
    plot_2vectors(dp_label, dp_prediction, 'dp')



