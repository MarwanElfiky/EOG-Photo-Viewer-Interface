import os
import matplotlib.pyplot as plt
import numpy as np
from pywt import wavedec
from scipy.signal import butter, filtfilt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
                    signal = [int(line.strip()) for line in f]
                    h_signals.append(signal)
            elif file.endswith('v.txt'):
                with open(os.path.join(class_path, file), 'r') as f:
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
        model = AutoReg(l, lags=16)
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
    max_coeffs_length = 0
    
    # First pass to find the maximum length
    for l in signal:
        y = wavedec(data=l, wavelet='db3', level=6)
        coeffs = list(y[1]) + list(y[2]) + list(y[3]) + list(y[4])
        max_coeffs_length = max(max_coeffs_length, len(coeffs))

    
    # Second pass to create vectors of uniform length
    for l in signal:
        y = wavedec(data=l, wavelet='db3', level=6)
        coeffs = list(y[1]) + list(y[2]) + list(y[3]) + list(y[4])

        # Pad with zeros if necessary to make all feature vectors the same length
        if len(coeffs) < max_coeffs_length:
            coeffs = coeffs + [0] * (max_coeffs_length - len(coeffs))
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


def merge_horizontal_vertical_features(h_v_features):
    merged_features = {}
    
    for cls in h_v_features['horizontal'].keys():
        merged_features[cls] = []
        
        # Make sure we have same number of samples in both directions
        min_samples = min(len(h_v_features['horizontal'][cls]), len(h_v_features['vertical'][cls]))
        
        for i in range(min_samples):
            # Check if features exist in both directions
            if i < len(h_v_features['horizontal'][cls]) and i < len(h_v_features['vertical'][cls]):
                # Get horizontal and vertical features
                h_features = h_v_features['horizontal'][cls][i]
                v_features = h_v_features['vertical'][cls][i]
                
                h_features = list(h_features)
                v_features = list(v_features)
                
                combined = h_features + v_features
                merged_features[cls].append(combined)
    
    return merged_features


def svm_classifier(features_dict, kernel='rbf', c=10):
    x = []
    y = []
    
    # First, check if all feature vectors have the same length
    feature_lengths = set()
    for class_name, feature_sets in features_dict.items():
        for feature_set in feature_sets:
            feature_lengths.add(len(feature_set))
    
    if len(feature_lengths) > 1:
        print(f"Warning: Found inconsistent feature vector lengths: {feature_lengths}")
        
        # Find the maximum length and pad shorter vectors
        max_length = max(feature_lengths)
        print(f"Standardizing all feature vectors to length {max_length}")
        
        for class_name, feature_sets in features_dict.items():
            for i, feature_set in enumerate(feature_sets):
                if len(feature_set) < max_length:
                    # Pad with zeros
                    features_dict[class_name][i] = list(feature_set) + [0] * (max_length - len(feature_set))
    
    # Now create feature vectors and labels
    for class_name, feature_sets in features_dict.items():
        for feature_set in feature_sets:
            x.append(feature_set)
            y.append(class_name)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X = np.array(x)
    
    # Apply standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

    svm = SVC(kernel=kernel, C=c, gamma='scale', class_weight='balanced')
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # class_names = label_encoder.classes_
    # print(classification_report(y_test, y_pred, target_names=class_names))
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # disp.plot(cmap='Blues')
    # plt.title("Confusion Matrix")
    # plt.show()
    
    return svm, accuracy, label_encoder


def preprocess_data():
    eog_signal = read_paired_eog("X:/AH/24_25_2/HCI/Project/Dataset/class")
    
    filtered_signal = apply_bandpass_filter(eog_signal)
    
    dc_free_signal = apply_dc_removal(filtered_signal)
    
    normalized_signal = apply_normalization(dc_free_signal)
    
    # trimmed_signal = apply_adaptive_trimming(normalized_signal)
    
    return normalized_signal, eog_signal



preprocessed_data, eog = preprocess_data()

ar = extract_AR_coefficients(preprocessed_data)

wavelets = extract_wavelet_coeffs(preprocessed_data)

stats = extract_all_statistical_features(preprocessed_data)

combined_by_direction = combine_features_by_direction(ar, wavelets, stats)

merged_features = merge_horizontal_vertical_features(combined_by_direction)

h_v_ar = merge_horizontal_vertical_features({'horizontal': ar['horizontal'], 'vertical': ar['vertical']})
print("AR COEFFICIENTS MODEL ACCURACY")
svm_classifier(h_v_ar, 'rbf', 10)


h_v_wavelets = merge_horizontal_vertical_features({'horizontal': wavelets['horizontal'], 'vertical': wavelets['vertical']})
print("wavelets COEFFICIENTS MODEL ACCURACY")
svm_classifier(h_v_wavelets, 'rbf', 10)

h_v_stat = merge_horizontal_vertical_features({'horizontal': stats['horizontal'], 'vertical': stats['vertical']})
print("STATISTICAL COEFFICIENTS MODEL ACCURACY")
svm_classifier(h_v_stat, 'rbf', 10)

print("ALL FEATURES MODEL ACCURACY")
svm_classifier(merged_features, 'rbf', 10)
# optimize_svm(merged_features)



# def main():
#     # Preprocess data
#
#
#     # Extract features from both horizontal and vertical signals
#     print("Extracting AR coefficients...")
#
#
#     print("Extracting wavelet coefficients...")
#
#
#     print("Extracting statistical features...")
#
#
#     # 1. Combine features by direction first
#     print("Combining features by direction...")
#
#
#     # 2. Merge horizontal and vertical features
#     print("Merging horizontal and vertical features...")
#
#
#     # Classification with individual feature types (combined H+V)
#     print("\nClassifying with AR coefficients only (H+V)...")
#
#
#     print("\nClassifying with Wavelet coefficients only (H+V)...")
#
#
#     print("\nClassifying with Statistical features only (H+V)...")
#     h_v_stats = merge_horizontal_vertical_features({'horizontal': stats['horizontal'], 'vertical': stats['vertical']})
#     svm_classifier(h_v_stats, 'linear', 100)
#
#     # Classification with all combined features
#     print("\nClassifying with all combined features (H+V)...")
#     # model, accuracy, encoder = svm_classifier(merged_features, 'rbf', 10)
#
#     # Optimize model parameters
#     print("\nOptimizing model with all combined features...")
#     best_model, best_accuracy, best_encoder = optimize_svm(merged_features)
#
#     return merged_features, best_model, best_encoder
#
#
# if __name__ == "__main__":
#     features, model, encoder = main()
#     print("\nFeature dimensions:")
#     for cls, feature_sets in features.items():
#         if feature_sets:
#             print(f"{cls}: {len(feature_sets)} samples, {len(feature_sets[0])} features")
#
#     print(f"\nFinal model accuracy: {model[1] * 100:.2f}%")
