# -*- coding: utf-8 -*-
from prediction import *

data_file = "../data/simu_10000_0.1_141_178_test.npy"
prediction_file = '../data/prediction.npy'

dp_MAE, dp_prediction, sp_MAE, sp_prediction = prediction(data_file)
dp_prediction = dp_prediction.detach().numpy()
sp_prediction = sp_prediction.detach().numpy()
np.save(prediction_file, [sp_prediction, dp_prediction])

print('===============')
print('Systolic MAE:', sp_MAE)
print('===============')
print('Diastolic MAE:', dp_MAE)
print('===============')

data_set = np.load(data_file)
sp_label = data_set[:, -2]
dp_label = data_set[:, -1]

[sp_prediction, dp_prediction] = np.load(prediction_file)

plot_2vectors(sp_label, sp_prediction, 'sp')
plot_2vectors(dp_label, dp_prediction, 'dp')





