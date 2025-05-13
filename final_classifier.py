import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import lstsq
import pywt
from scipy.signal import butter, filtfilt, decimate, find_peaks
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import joblib


# IMPROVED DATA READING WITH SIGNAL QUALITY CHECK
def read_eog_data(folder_path, min_signal_length=100):
    """
    Reads EOG signal .txt files from organized class folders.
    Each class has its own folder containing 20 signal files.
    
    Args:
        folder_path (str): Path to the main data folder containing class subfolders
        min_signal_length (int): Minimum required length for valid signals
    
    Returns:
        dict: Dictionary with class names as keys and lists of signals as values
    """
    # Define class names (folder names)
    classes = ['up', 'down', 'right', 'left', 'blink']
    data = {cls: [] for cls in classes}
    discarded = {cls: 0 for cls in classes}

    # Process each class folder
    for cls in classes:
        class_folder = os.path.join(folder_path, cls)
        if not os.path.exists(class_folder):
            print(f"Warning: Class folder '{cls}' not found at {class_folder}")
            continue

        # Process each signal file in the class folder
        for filename in os.listdir(class_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(class_folder, filename)
                try:
                    with open(file_path, 'r') as f:
                        signal = [float(line.strip()) for line in f if line.strip()]

                    # Quality check - discard very short signals or those with extreme values
                    if len(signal) >= min_signal_length and np.std(signal) > 0.1 and np.abs(np.max(signal)) < 1000:
                        data[cls].append(signal)
                    else:
                        discarded[cls] += 1
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
                    discarded[cls] += 1

    # Print class distribution
    print("\nClass distribution:")
    for cls in classes:
        print(f"{cls}: {len(data[cls])} samples (discarded: {discarded[cls]})")

    return data


# Enhanced preprocessing with additional steps
def preprocess_eog_data(data_dict, fs=250, lowcut=0.1, highcut=40):
    """
    Advanced preprocessing with adaptive signal trimming and multiple filters
    """
    processed = {}
    for cls, signals in data_dict.items():
        processed[cls] = []
        for sig in signals:
            # 1. Convert to numpy array
            signal = np.array(sig)

            # 2. Handle NaN and Inf values
            signal = np.nan_to_num(signal, nan=0.0, posinf=np.max(signal[~np.isinf(signal)]),
                                   neginf=np.min(signal[~np.isinf(signal)]))

            # 3. Apply bandpass filter
            filtered = bandpass_filter(signal, lowcut, highcut, fs, order=6)

            # 4. Apply adaptive signal trimming - keep most active region
            trimmed = adaptive_trim_signal(filtered)

            # 5. Standardize signal length - important for wavelet features
            resampled = standardize_signal_length(trimmed, target_length=1000)

            # 6. Normalize
            normalized = normalize_signal(resampled)

            processed[cls].append(normalized)
    return processed


def bandpass_filter(signal, lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def normalize_signal(signal):
    signal = np.array(signal)
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-10)  # Add small epsilon to avoid division by zero


def adaptive_trim_signal(signal, window_size=100):
    """Trim signal adaptively based on activity level"""
    if len(signal) <= window_size:
        return signal

    # Compute activity (variance) in sliding windows
    activity = []
    for i in range(len(signal) - window_size):
        activity.append(np.var(signal[i:i + window_size]))

    activity = np.array(activity)

    # Find peak activity regions
    threshold = np.mean(activity) + 0.5 * np.std(activity)
    active_regions = activity > threshold

    if np.sum(active_regions) > window_size:
        # Find start and end of most active region
        starts = np.where(np.diff(np.concatenate(([0], active_regions))) > 0)[0]
        ends = np.where(np.diff(np.concatenate((active_regions, [0]))) < 0)[0]

        if len(starts) > 0 and len(ends) > 0:
            # Find longest active region
            lengths = ends - starts
            longest_idx = np.argmax(lengths)
            start = max(0, starts[longest_idx] - window_size // 2)
            end = min(len(signal), ends[longest_idx] + window_size // 2)
            return signal[start:end]

    # Fallback to simple trimming if adaptive approach fails
    start = max(0, len(signal) // 4)
    end = min(len(signal), 3 * len(signal) // 4)
    return signal[start:end]


def standardize_signal_length(signal, target_length=1000):
    """Resample signal to standard length"""
    current_length = len(signal)
    if current_length == target_length:
        return signal

    # Simple linear interpolation
    indices = np.linspace(0, current_length - 1, target_length)
    return np.interp(indices, np.arange(current_length), signal)


# ADVANCED FEATURE EXTRACTION

def extract_time_domain_features(signal):
    """Extract comprehensive time domain features"""
    features = []

    # Basic statistics
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.min(signal))
    features.append(np.max(signal))
    features.append(np.max(signal) - np.min(signal))  # Range
    features.append(np.median(signal))

    # Percentiles
    for p in [10, 25, 75, 90]:
        features.append(np.percentile(signal, p))

    # Higher order statistics
    features.append(np.mean(np.power(signal - np.mean(signal), 3)))  # Skewness-like
    features.append(np.mean(np.power(signal - np.mean(signal), 4)))  # Kurtosis-like

    # Signal dynamics
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    features.append(np.mean(np.abs(diff1)))
    features.append(np.std(diff1))
    features.append(np.mean(np.abs(diff2)))
    features.append(np.std(diff2))

    # Zero crossings and peaks
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    features.append(len(zero_crossings))

    # Find peaks (both positive and negative)
    peaks, _ = find_peaks(signal, height=np.std(signal) / 2)
    neg_peaks, _ = find_peaks(-signal, height=np.std(signal) / 2)
    features.append(len(peaks))
    features.append(len(neg_peaks))

    # Mean peak amplitude 
    if len(peaks) > 0:
        features.append(np.mean(signal[peaks]))
    else:
        features.append(0)

    if len(neg_peaks) > 0:
        features.append(np.mean(signal[neg_peaks]))
    else:
        features.append(0)

    return np.array(features)


def extract_frequency_domain_features(signal):
    """Extract frequency domain features"""
    # Compute FFT
    fft = np.abs(np.fft.rfft(signal))
    fft_freq = np.fft.rfftfreq(len(signal))

    # Normalize
    fft = fft / np.sum(fft)

    features = []

    # Power in different frequency bands (normalized)
    bands = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
    for low, high in bands:
        mask = (fft_freq >= low) & (fft_freq < high)
        features.append(np.sum(fft[mask]))

    # Dominant frequency
    features.append(fft_freq[np.argmax(fft)])

    # Spectral statistics
    features.append(np.mean(fft))
    features.append(np.std(fft))
    features.append(np.max(fft))

    # Spectral entropy
    entropy = -np.sum(fft * np.log2(fft + 1e-10))
    features.append(entropy)

    return np.array(features)


def extract_wavelet_stats(signal, wavelet='db4', level=5):
    """Extract wavelet-based features with statistical measures"""
    # Perform wavelet decomposition
    try:
        coeffs = pywt.wavedec(signal, wavelet,
                              level=min(level, pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)))
    except Exception:
        # Fallback if error occurs
        coeffs = pywt.wavedec(signal, wavelet, level=3)

    features = []

    # Extract statistics from each coefficient level
    for i, coeff in enumerate(coeffs):
        features.append(np.mean(coeff))
        features.append(np.std(coeff))
        features.append(np.max(np.abs(coeff)))
        features.append(np.sum(coeff ** 2))  # Energy

        # Zero crossing rate
        if len(coeff) > 1:
            features.append(np.sum(np.diff(np.signbit(coeff)) != 0) / len(coeff))
        else:
            features.append(0)

    return np.array(features)


def extract_ar_coeffs(signal, order=12):
    """Extract AR coefficients with higher order"""
    x = np.array(signal)
    N = len(x)
    if N <= order:
        return np.zeros(order)  # Return zeros for too short signals

    try:
        X = np.column_stack([x[i:N - order + i] for i in range(order)])
        y = x[order:]
        coeffs, _, _, _ = lstsq(X, y, rcond=None)
        return coeffs
    except Exception:
        # Fallback to zeros if error occurs
        return np.zeros(order)


def extract_super_features(signal):
    """Extract a comprehensive set of features combining multiple domains"""
    time_features = extract_time_domain_features(signal)
    freq_features = extract_frequency_domain_features(signal)
    wavelet_features = extract_wavelet_stats(signal, wavelet='db4', level=4)
    ar_features = extract_ar_coeffs(signal, order=12)

    # Combine all features
    return np.concatenate([time_features, freq_features, wavelet_features, ar_features])


def extract_features(data_dict, method='super'):
    """Extract features using the specified method"""
    features = {}
    for cls, signals in data_dict.items():
        features[cls] = []
        for sig in signals:
            if method == 'ar':
                feat = extract_ar_coeffs(sig, order=12)
            elif method == 'wavelet':
                feat = extract_wavelet_stats(sig)
            elif method == 'time':
                feat = extract_time_domain_features(sig)
            elif method == 'frequency':
                feat = extract_frequency_domain_features(sig)
            elif method == 'super':
                feat = extract_super_features(sig)
            else:
                raise ValueError('Unknown method')
            features[cls].append(feat)
    return features


def prepare_data(features_dict):
    """Convert features dict to X (features) and y (labels) arrays"""
    X = []
    y = []
    for cls, feats in features_dict.items():
        for f in feats:
            X.append(f)
            y.append(cls)
    return np.array(X), np.array(y)


def train_and_evaluate_model(X, y, test_size=0.2, random_state=42):
    """Train and evaluate the SVM model with feature selection"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Create pipeline with feature selection and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=min(X.shape[1] // 2, 50))),  # Select top 50 features
        ('classifier', SVC(probability=True))
    ])

    # Define parameter grid
    param_grid = {
        'classifier__C': [1, 10, 100],
        'classifier__gamma': ['scale', 0.01, 0.1],
        'classifier__kernel': ['rbf', 'poly'],
        'feature_selection__k': [min(X.shape[1] // 2, 50), min(X.shape[1] // 3, 30), min(X.shape[1] // 4, 20)]
    }

    # Find best parameters
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train_enc)

    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Cross-validation score
    cv_score = cross_val_score(best_model, X_train, y_train_enc, cv=cv)
    cv_mean = cv_score.mean()

    # Test set evaluation
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test_enc, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test_enc, y_pred)

    # Return results
    return {
        'model': best_model,
        'encoder': le,
        'best_params': best_params,
        'cv_score': cv_mean,
        'test_accuracy': test_acc,
        'confusion_matrix': cm,
        'class_names': le.classes_
    }


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()


# Main execution
if __name__ == "__main__":
    data_path = "X:/AH/24_25_2/HCI/Project/Dataset/class"

    # Step 1: Read and preprocess data
    print("\n===== Reading Data =====")
    data = read_eog_data(data_path)

    print("\n===== Preprocessing Data =====")
    preprocessed_data = preprocess_eog_data(data, fs=250, lowcut=0.5, highcut=20)

    # Step 2: Extract features
    print("\n===== Extracting Super Features =====")
    features = extract_features(preprocessed_data, method='super')
    X, y = prepare_data(features)
    print(f"Feature vector size: {X.shape[1]} features")

    # Step 3: Train and evaluate model
    print("\n===== Training and Evaluating Model =====")
    results = train_and_evaluate_model(X, y)

    # Step 4: Report results
    print("\n===== Results =====")
    print(f"Cross-validation accuracy: {results['cv_score']:.4f}")
    print(f"Test set accuracy: {results['test_accuracy']:.4f}")
    print(f"Best parameters: {results['best_params']}")

    # Step 5: Plot confusion matrix
    print("\n===== Generating Confusion Matrix =====")
    plot_confusion_matrix(results['confusion_matrix'], results['class_names'])
    print("Confusion matrix saved to 'confusion_matrix.png'")

    # Step 6: Save the model
    joblib.dump(results['model'], 'eog_classifier_model.pkl')
    joblib.dump(results['encoder'], 'label_encoder.pkl')
    print("Model saved to 'eog_classifier_model.pkl'")
