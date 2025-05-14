import os
import matplotlib.pyplot as plt
import numpy as np
from pywt import wavedec
from scipy.signal import butter, filtfilt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler


def read_eog(folder_path):
    # Number of classes : 5 : ['Blink', 'Down', 'Left', 'Right', 'Up']
    classes = os.listdir(folder_path)
    class_data = {}
    for folder in classes:
        class_path = os.path.join(folder_path, folder)
        files = (os.listdir(class_path))
        signals = []
        for file in files:
            file_path = os.path.join(class_path, file)
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
    for cls, signal_list in signal_dict.items():
        filtered_signal_listOflists = bandpass_butter_filter(signal_list)
        filtered_signal_dict[cls] = filtered_signal_listOflists
    return filtered_signal_dict


def normalize_021(signals):
    normalized_signal = []
    for l in signals:
        min_value = np.min(l)
        max_value = np.max(l)
        normalized_signal.append([(i - min_value) / (max_value - min_value) for i in l])
    return normalized_signal


def dc_component_removal(signal):
    removed_mean_signal = []
    for l in signal:
        mean_for_list = np.mean(l)
        removed_mean_signal.append([i - mean_for_list for i in l])
    return removed_mean_signal


def apply_dc_removal(normalized_signal_dict):
    signal_after_dc_removal = {}
    for cls, signal_list in normalized_signal_dict.items():
        new_signal_listOflists = dc_component_removal(signal_list)
        signal_after_dc_removal[cls] = new_signal_listOflists
    return signal_after_dc_removal


def apply_normalization(filtered_signal_dict):
    normalized_signal_dict = {}
    for cls, signal_list in filtered_signal_dict.items():
        normalized_signal_listOflists = normalize_021(signal_list)
        normalized_signal_dict[cls] = normalized_signal_listOflists
    return normalized_signal_dict



def ar_coeffs_calc(signal_listOflists):
    extracted_coeffs = []
    for l in signal_listOflists:
        model = AutoReg(l, lags=16)
        model_fit = model.fit()
        ar_coeffs = model_fit.params
        extracted_coeffs.append(list(ar_coeffs))
    return extracted_coeffs


def extract_AR_coefficients(normalized_signal_dict):
    ar_coeffs_dict = {}
    for cls, signal_list in normalized_signal_dict.items():
        extracted_ar_lols = ar_coeffs_calc(signal_list)
        ar_coeffs_dict[cls] = extracted_ar_lols
    return ar_coeffs_dict


def wavelet_coeffs(signal):
    decomposed = []
    for l in signal:
        y = wavedec(data=l, wavelet='db3', level=6)
        decomposed.append(list(y[1]) + list(y[2]) + list(y[3]) + list(y[4]))
    return decomposed


def extract_wavelet_coeffs(normalized_signal_dict):
    extracted_wavelet_coeffs = {}
    for cls, signal in normalized_signal_dict.items():
        extracted_wavelet_lols = wavelet_coeffs(signal)
        extracted_wavelet_coeffs[cls] = extracted_wavelet_lols
    return extracted_wavelet_coeffs


def svm_classifier(features_dict: dict, kernel, c):
    x = []
    y = []
    for class_name, feature_sets in features_dict.items():
        for feature_set in feature_sets:
            x.append(feature_set)
            y.append(class_name)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X = np.array(x)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    svm = SVC(kernel=kernel, C=c, gamma='scale', class_weight='balanced')
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    class_names = label_encoder.classes_
    # print(classification_report(y_test, y_pred, target_names=class_names))
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # disp.plot(cmap='Blues')
    # plt.title("Confusion Matrix")
    # plt.show()
    return svm, accuracy



def preprocess_data():
    eog_signal = read_eog("X:/AH/24_25_2/HCI/Project/Dataset/class")
    filtered_signal = apply_bandpass_filter(eog_signal)
    dc_free_signal = apply_dc_removal(filtered_signal)
    final_normalized_data = apply_normalization(dc_free_signal)
    return final_normalized_data


preprocessed_data = preprocess_data()
ar = extract_AR_coefficients(preprocessed_data)
wavelets = extract_wavelet_coeffs(preprocessed_data)


svm_classifier(ar, 'rbf', 10)
svm_classifier(wavelets, 'rbf', 10)


