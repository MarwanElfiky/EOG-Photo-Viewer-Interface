import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt


def read_eog(folder_path):
    # Number of classes : 5 : ['Blink', 'Down', 'Left', 'Right', 'Up']
    classes = os.listdir(folder_path)
    class_data = {}
    for folder in classes:
        # Class Path = Each class path
        class_path = os.path.join(folder_path, folder)
        files = (os.listdir(class_path))
        signals = []
        for file in files:
            file_path = os.path.join(class_path, file)
            # print(file_path)
            with open(file_path, 'r') as f:
                signal = [int(line.strip()) for line in f]
                signals.append(signal)
        class_data[folder] = signals
    return class_data


def plot_signal(signal):
    plt.figure(figsize=(12, 6))
    plt.xlabel("time")
    plt.ylabel("Amplitudes")
    # print(len(np.arange(0, len(signal))))
    plt.plot(np.arange(0, len(signal)), signal)
    plt.show()


def bandpass_butter_filter(signal, fs=250, lowcut=1, highcut=20, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    filtered = []
    b, a = butter(order, [low, high], btype='bandpass')
    for l in signal:
        y = filtfilt(b, a, l)
        y = list(y)
        filtered.append(y)
    return filtered


def apply_bandpass_filter(signal_dict):
    filtered_signal_dict = {}
    # apply BPF 5 times one for each class
    for cls, signal_list in signal_dict.items():
        filtered_signal_listOflists = bandpass_butter_filter(signal_list)
        filtered_signal_dict[cls] = filtered_signal_listOflists
    return filtered_signal_dict


eog_signal = read_eog("X:/AH/24_25_2/HCI/Project/Dataset/class")
filtered_signal = apply_bandpass_filter(eog_signal)
print(eog_signal['Left'][0])
print(filtered_signal['Left'][0])
plot_signal(eog_signal['Left'][0])
plot_signal(filtered_signal['Left'][0])
