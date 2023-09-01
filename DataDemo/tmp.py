import numpy as np
from scipy.signal import hilbert, chirp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema, find_peaks
from scipy.interpolate import interp1d
import datasim as nk


def update_array(a, data_tmp):
    i = 0
    while i < len(a) - 2:
        if data_tmp[a[i]] < data_tmp[a[i + 1]] < data_tmp[a[i + 2]]:
            a = np.delete(a, i)
        elif data_tmp[a[i]] > data_tmp[a[i + 1]] > data_tmp[a[i + 2]]:
            a = np.delete(a, i + 2)
        else:
            i += 1
    return a


# def delete_unique(a):
#     mean = np.mean(a)
#     std = np.std(a)
#     # 设置阈值
#     threshold = 1
#     # 使用布尔索引删除特殊值
#     filtered_a = a[np.abs(a - mean) <= threshold * std]
#     return filtered_a

def delete_unique(arr):
    counts = np.bincount(arr)
    most_frequent = np.argmax(counts)
    return arr[arr == most_frequent]


def envelope_hilbert(signal, fs):
    z = hilbert(signal)  # form the analytical signal
    inst_amplitude = np.abs(z)  # envelope extraction
    inst_phase = np.unwrap(np.angle(z))  # inst phase
    inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs  # inst frequency

    # Regenerate the carrier from the instantaneous phase
    regenerated_carrier = np.cos(inst_phase)

    return inst_amplitude, inst_freq, inst_phase, regenerated_carrier


hilbert_envelope_D = [0, 0]

data_test = np.load("../data/simu_10000_0.1_141_178_test.npy")
data_train = np.load("../data/simu_20000_0.1_90_140_train.npy")
fs = 100
t = np.linspace(0, 10, 10 * fs)

for i in range(1):
    signal = data_test[i, :1000]
    inst_amplitude, inst_freq, inst_phase, regenerated_carrier = envelope_hilbert(signal, 100)

    window_size = 3  # Adjust the window size as needed
    smoothed_envelope = np.convolve(inst_amplitude, np.ones(window_size) / window_size, mode='same')

    plt.plot(signal)
    plt.plot(smoothed_envelope, 'r')  # overlay the extracted envelope
    plt.title('Modulated signal and extracted envelope')
    # plt.xlim(0, 200)
    plt.show()

    signal1 = signal
    signal = smoothed_envelope

    peak_indices, _ = find_peaks(signal)  # 返回极大值点的索引
    # 线性插值
    t_peaks = t[peak_indices]  # 极大值点的时间
    peak_values = signal[peak_indices]  # 极大值点的幅值
    interpolation_func = interp1d(t_peaks, peak_values, kind='linear', bounds_error=False, fill_value=0)
    envelope = interpolation_func(t)

    peaks2, _ = find_peaks(envelope, distance=10)

    peaks2 = update_array(peaks2, signal)
    if len(peaks2) % 2 != 0:
        peaks2 = np.delete(peaks2, len(peaks2) - 1)

    #     print(peaks2[0])

    diff_12 = peaks2[1::2] - peaks2[0::2]
    diff_21 = peaks2[2::2] - peaks2[1:-1:2]
    #     diff_22 = peaks2[2::2] - peaks2[:-2:2]

    if len(diff_12) == 0:
        continue

    if len(diff_21) == 0:
        continue

    diff_12 = delete_unique(diff_12)
    diff_21 = delete_unique(diff_21)
    #     diff_22 = delete_unique(diff_22)

    #     print(diff_12)
    #     print(diff_21)

    D12 = np.mean(diff_12)
    D21 = np.mean(diff_21)
    tmp = np.array([D12, D21])
    hilbert_envelope_D = np.vstack((hilbert_envelope_D, tmp))
    print(D12, D21)

    info = nk.scg_findpeaks(signal1,sampling_rate=100)
    nk.events_plot(info["scg_R_Peaks"],signal1)
    plt.show()





# print(hilbert_envelope_D.shape)
# print(hilbert_envelope_D)