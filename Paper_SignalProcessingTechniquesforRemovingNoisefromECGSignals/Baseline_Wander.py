import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# Sampling Frequency
Fs = 100
# Order
N = 50
# Cutoff Frequency
Fc = 0.1

# Function to design Butterworth filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Function to apply filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Load the ECG signal
x = np.load("../data/simu_10000_0.1_141_178_test.npy")
x1 = x[555, :1000]
x2 = x1 / np.max(x1)

# Plot ECG Signal with low-frequency (baseline wander) noise
plt.subplot(2, 1, 1)
plt.plot(x1)
plt.title('ECG Signal with low-frequency (baseline wander) noise')
plt.grid(True)

# Apply filter
y0 = butter_lowpass_filter(x2, Fc, Fs, N)

# Plot ECG signal with baseline wander REMOVED

plt.subplot(2, 1, 2)
plt.plot(y0)
plt.title('ECG signal with baseline wander REMOVED')
plt.grid(True)

plt.show()
