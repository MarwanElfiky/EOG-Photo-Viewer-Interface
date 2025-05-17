import os
from typing import Any
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pywt import wavedec
from scipy.signal import butter, filtfilt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
import openpyxl
from openpyxl import Workbook


def read_paired_eog(folder_path):
    classes = os.listdir(folder_path)
    class_data = {'horizontal': {}, 'vertical': {}}

    for folder in classes:
        class_path = os.path.join(folder_path, folder)
        files = os.listdir(class_path)
        h_signals = []
        v_signals = []

        for file in files:
            if file.endswith('h.txt'):
                with open(os.path.join(class_path, file), 'r') as f:
                    # print(file)
                    signal = [int(line.strip()) for line in f]
                    h_signals.append(signal)
            elif file.endswith('v.txt'):
                with open(os.path.join(class_path, file), 'r') as f:
                    # print(file)
                    signal = [int(line.strip()) for line in f]
                    v_signals.append(signal)

        class_data['horizontal'][folder] = h_signals
        class_data['vertical'][folder] = v_signals
    return class_data


def plot_signal(signal, title="Signal Plot"):
    plt.figure(figsize=(12, 6))
    plt.xlabel("time")
    plt.ylabel("Amplitudes")
    plt.title(title)
    plt.plot(np.arange(0, len(signal)), signal)
    plt.show()


def bandpass_butter_filter(signal, fs=176, lowcut=1, highcut=22, order=2):
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
    filtered_signal_dict = {'horizontal': {}, 'vertical': {}}
    for direction in ['horizontal', 'vertical']:
        for cls, signal_list in signal_dict[direction].items():
            filtered_signal_listOflists = bandpass_butter_filter(signal_list)
            filtered_signal_dict[direction][cls] = filtered_signal_listOflists
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
    signal_after_dc_removal = {'horizontal': {}, 'vertical': {}}
    for direction in ['horizontal', 'vertical']:
        for cls, signal_list in normalized_signal_dict[direction].items():
            new_signal_listOflists = dc_component_removal(signal_list)
            signal_after_dc_removal[direction][cls] = new_signal_listOflists
    return signal_after_dc_removal


def apply_normalization(filtered_signal_dict):
    normalized_signal_dict = {'horizontal': {}, 'vertical': {}}
    for direction in ['horizontal', 'vertical']:
        for cls, signal_list in filtered_signal_dict[direction].items():
            normalized_signal_listOflists = normalize_021(signal_list)
            normalized_signal_dict[direction][cls] = normalized_signal_listOflists
    
    return normalized_signal_dict



def ar_coeffs_calc(signal_listOflists):
    extracted_coeffs = []
    for l in signal_listOflists:
        model = AutoReg(l, lags=4)
        model_fit = model.fit()
        ar_coeffs = model_fit.params
        extracted_coeffs.append(list(ar_coeffs))
    return extracted_coeffs


def extract_AR_coefficients(normalized_signal_dict):
    ar_coeffs_dict = {'horizontal': {}, 'vertical': {}}
    
    for direction in ['horizontal', 'vertical']:
        for cls, signal_list in normalized_signal_dict[direction].items():
            extracted_ar_lols = ar_coeffs_calc(signal_list)
            ar_coeffs_dict[direction][cls] = extracted_ar_lols
    
    return ar_coeffs_dict


def wavelet_coeffs(signal):
    decomposed = []
    for l in signal:
        y = wavedec(data=l, wavelet='db3', level=2)
        coeffs = list(y[0])
        decomposed.append(coeffs)
    return decomposed


def extract_wavelet_coeffs(normalized_signal_dict):
    extracted_wavelet_coeffs = {'horizontal': {}, 'vertical': {}}
    
    for direction in ['horizontal', 'vertical']:
        for cls, signal in normalized_signal_dict[direction].items():
            extracted_wavelet_lols = wavelet_coeffs(signal)
            extracted_wavelet_coeffs[direction][cls] = extracted_wavelet_lols
    
    return extracted_wavelet_coeffs


def extract_statistical_features(signal):
    features = []
    for s in signal:
        mean = np.mean(s)
        std = np.std(s)
        variance = np.var(s)
        energy = np.sum(np.square(s))
        
        stat_features = [mean, std, variance, energy]
        features.append(stat_features)
    
    return features


def extract_all_statistical_features(normalized_signal_dict):
    stat_features_dict = {'horizontal': {}, 'vertical': {}}
    
    for direction in ['horizontal', 'vertical']:
        for cls, signal_list in normalized_signal_dict[direction].items():
            extracted_stat_features = extract_statistical_features(signal_list)
            stat_features_dict[direction][cls] = extracted_stat_features
    
    return stat_features_dict


def combine_features_by_direction(ar, wavelets, stats):
    combined = {'horizontal': {}, 'vertical': {}}
    
    for direction in ['horizontal', 'vertical']:
        for cls in ar[direction].keys():
            combined[direction][cls] = []
            for i in range(len(ar[direction][cls])):
                feature_vector = ar[direction][cls][i] + wavelets[direction][cls][i] + stats[direction][cls][i]
                combined[direction][cls].append(feature_vector)
    
    return combined


def combine_single_feature_by_direction(feature_dict):
    combined = {}

    for direction in ['horizontal', 'vertical']:
        for cls, feature_lists in feature_dict[direction].items():
            if cls not in combined:
                combined[cls] = []
            for feature_vector in feature_lists:
                combined[cls].append(feature_vector)

    return combined



def merge_horizontal_vertical_features(h_v_features):
    merged_features = {}
    
    for cls in h_v_features['horizontal'].keys():
        merged_features[cls] = []
        
        min_samples = min(len(h_v_features['horizontal'][cls]), len(h_v_features['vertical'][cls]))
        
        for i in range(min_samples):
            if i < len(h_v_features['horizontal'][cls]) and i < len(h_v_features['vertical'][cls]):
                h_features = h_v_features['horizontal'][cls][i]
                v_features = h_v_features['vertical'][cls][i]
                
                h_features = list(h_features)
                v_features = list(v_features)
                
                combined = h_features + v_features
                merged_features[cls].append(combined)
    
    return merged_features


def svm_classifier(features_dict, c, kernel='rbf'):
    # print(len(features_dict['Blink'][0]))
    x = []
    y = []

    for cls, feature_sets in features_dict.items():
        for feature_set in feature_sets:
            x.append(feature_set)
            y.append(cls)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X = np.array(x)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

    svm = SVC(kernel=kernel, C=c)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    train_accuracy = accuracy_score(y_train, svm.predict(X_train))
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    # class_names = label_encoder.classes_
    # print(classification_report(y_test, y_pred, target_names=class_names))
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # disp.plot(cmap='Blues')
    # plt.title("Confusion Matrix")
    # plt.show()
    # joblib.dump(svm, model_name)
    return svm, label_encoder


def preprocess_data():
    eog_signal = read_paired_eog("X:/AH/24_25_2/HCI/Project/Dataset/class")
    filtered_signal = apply_bandpass_filter(eog_signal)
    dc_free_signal = apply_dc_removal(filtered_signal)
    normalized_signal = apply_normalization(dc_free_signal)

    return normalized_signal, eog_signal


def save_eog_to_excel(eog: dict[str, dict], filename: str):
    wb = Workbook()
    ws = wb.active
    ws.title = "EOG_Data"

    max_len = 0
    for direction in eog:
        for class_name in eog[direction]:
            for signal in eog[direction][class_name]:
                max_len = max(max_len, len(signal))

    headers = ['Direction', 'Class', 'Sample_Number'] + [f'Value_{i + 1}' for i in range(max_len)]
    ws.append(headers)

    for direction in ['horizontal', 'vertical']:
        for class_name, signals in eog[direction].items():
            for i, signal_data in enumerate(signals):
                row_data = [direction, class_name, i + 1] + signal_data
                ws.append(row_data)

    wb.save(filename if filename.endswith('.xlsx') else filename + '.xlsx')


def save_merged_features_to_excel(merged_features: dict[Any, list], filename: str):
    wb = Workbook()
    ws = wb.active
    ws.title = "Merged_Features"

    max_len = 0
    for class_name in merged_features:
        for feature_vector in merged_features[class_name]:
            max_len = max(max_len, len(feature_vector))

    headers = ['Class', 'Sample_Number'] + [f'Feature_{i + 1}' for i in range(max_len)]
    ws.append(headers)

    for class_name, feature_vectors in merged_features.items():
        for i, feature_vector in enumerate(feature_vectors):
            row_data = [class_name, i + 1] + list(feature_vector)
            ws.append(row_data)

    wb.save(filename if filename.endswith('.xlsx') else filename + '.xlsx')


def preprocess_single_signal(h_signal, v_signal):
    temp_dict = {'horizontal': {}, 'vertical': {}}
    temp_dict['horizontal']['temp_class'] = [h_signal]
    temp_dict['vertical']['temp_class'] = [v_signal]

    filtered_dict = apply_bandpass_filter(temp_dict)

    dc_removed_dict = apply_dc_removal(filtered_dict)
    normalized_dict = apply_normalization(dc_removed_dict)

    return normalized_dict


def extract_features_single_signal(preprocessed_temp_dict: dict):

    ar_coeffs = extract_AR_coefficients(preprocessed_temp_dict)
    wavelet_coeffs = extract_wavelet_coeffs(preprocessed_temp_dict)
    stat_features = extract_all_statistical_features(preprocessed_temp_dict)

    combined = combine_features_by_direction(ar_coeffs, wavelet_coeffs, stat_features)

    merged_features = merge_horizontal_vertical_features(combined)

    return merged_features['temp_class'][0]



def work():
    preprocessed_data, eog = preprocess_data()

    ar = extract_AR_coefficients(preprocessed_data)

    wavelets = extract_wavelet_coeffs(preprocessed_data)


    stats = extract_all_statistical_features(preprocessed_data)


    combined_by_direction = combine_features_by_direction(ar, wavelets, stats)

    merged_features = merge_horizontal_vertical_features(combined_by_direction)
    # print(len(merged_features['Blink'][0]))

    # combined_ar = combine_single_feature_by_direction(ar)
    # print("SVM using AR features")
    # svm_classifier(combined_ar, 'rbf', 5)
    # combined_wave = combine_single_feature_by_direction(wavelets)
    # print("\nSVM using wavelets features")
    # svm_classifier(combined_wave, 'rbf', 5)
    # combined_stat = combine_single_feature_by_direction(stats)
    # print("\nSVM using statistical features")
    # svm_classifier(combined_stat, 'rbf', 5)
    # print("\nSVM using combined features")
    svm, label_encoder = svm_classifier(merged_features, 5, 'rbf')


    # joblib.dump(svm, 'svm_model.pkl')
    # joblib.dump(label_encoder, "label_encoder.pkl")


if __name__ == "__main__":
    work()
