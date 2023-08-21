from model import *
from utils import *
from dataset import *

def prediction(data_file):

    criterion = nn.L1Loss()

    loss_total = 0
    batch_num = 0

    # 预测S
    model = torch.load("pth/S_model_50_1.9573.pth", map_location=torch.device('cpu'))
    # model = torch.load("./pth/S_model_490_15.5063.pth")
    dataset_test = Dataset(data_file, 0, 1)
    train_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)

    tmp = np.load("./data/simu_20000_0.1_90_140_train.npy")
    max = np.max(tmp[:, 1004])
    min = np.min(tmp[:, 1004])

    loss_total = 0
    batch_num = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(), target.cuda()
        output = model(data)

        # output = output * (max - min) + min

        if 'sp_prediction' in locals():
            sp_prediction = torch.cat((sp_prediction, output), dim=0)
        else:
            sp_prediction = output

        loss = criterion(output, target)
        # print(loss.mean().item())
        loss_total = loss_total + loss.item()
        batch_num = batch_num + 1

    sp_prediction = sp_prediction.reshape(-1)
    sp_MAE = loss_total / batch_num



    return sp_MAE, sp_prediction

if __name__ == "__main__":
    data_file = "./data/test_without_resp.npy"
    prediction_file = './data/prediction.npy'

    sp_MAE, sp_prediction = prediction(data_file)
    sp_prediction = sp_prediction.detach().numpy()
    np.save(prediction_file, [sp_prediction])

    print('===============')
    print('Systolic MAE:', sp_MAE)
    print('===============')

    data_set = np.load(data_file)
    sp_label = data_set[:, -2]

    [sp_prediction] = np.load(prediction_file)

    plot_2vectors(sp_label, sp_prediction, 'sp')




