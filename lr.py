import os
import zipfile
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import neurokit2 as nk
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl

# 1. Uploads and extracts a ZIP folder containing multiple ECG CSV files.
zip_file_path = r'C:\CCP2\logisticregression\ecg_dataset_final.zip'
extract_dir = 'extracted_files'

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# 2. Reads each CSV, extracts the 'Time' and 'EcgWaveform' columns.
def read_ecg_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['Time', 'EcgWaveform']  # Set the column names
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['EcgWaveform'] = pd.to_numeric(df['EcgWaveform'], errors='coerce')
    df.dropna(inplace=True)
    return df[['Time', 'EcgWaveform']]

# 3. Preprocesses the ECG signal
def preprocess_ecg(time, ecg):
    if time.empty or ecg.empty:
        return None

    # Bandpass filtering (0.5–40 Hz)
    def bandpass_filter(signal, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    try:
        fs = int(1 / np.mean(np.diff(time)))  # Sampling frequency
        ecg_filtered = bandpass_filter(ecg, 0.5, 40, fs)

        # Baseline correction
        ecg_corrected = nk.ecg_clean(ecg_filtered, sampling_rate=fs)

        # R-peak detection
        _, rpeaks = nk.ecg_peaks(ecg_corrected, sampling_rate=fs)

        if len(rpeaks['ECG_R_Peaks']) < 2:
            return None

        # RR interval extraction
        rr_intervals = np.diff(rpeaks['ECG_R_Peaks']) / fs * 1000  # in milliseconds

        return rr_intervals if len(rr_intervals) > 0 else None
    except Exception as e:
        print(f"Error during ECG preprocessing: {e}")
        return None

# 4. Computes HRV features
def compute_hrv_features(rr_intervals):
    if rr_intervals is None or len(rr_intervals) == 0:
        return None, None, None, None

    sdnn = np.std(rr_intervals, ddof=1)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    mean_rr = np.mean(rr_intervals)
    median_rr = np.median(rr_intervals)
    return sdnn, rmssd, mean_rr, median_rr

# 5. Labels the dataset
def label_data(sdnn, rmssd):
    if sdnn is None or rmssd is None:
        return None
    return int(sdnn < 17.13 or rmssd < 24.94)  # 1 for CAN detected, 0 otherwise

# Process all files and create dataset
data = []

for root, _, files in os.walk(extract_dir):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            try:
                df = read_ecg_csv(file_path)
                if df.empty:
                    print(f"No valid data in {file_path}")
                    continue

                rr_intervals = preprocess_ecg(df['Time'], df['EcgWaveform'])
                if rr_intervals is None:
                    print(f"Insufficient RR intervals in {file_path}")
                    continue

                sdnn, rmssd, mean_rr, median_rr = compute_hrv_features(rr_intervals)
                label = label_data(sdnn, rmssd)
                if sdnn is not None and rmssd is not None and mean_rr is not None and median_rr is not None and label is not None:
                    data.append([sdnn, rmssd, mean_rr, median_rr, label])
                else:
                    print(f"Insufficient data in {file_path} for HRV feature extraction")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

columns = ['SDNN', 'RMSSD', 'Mean RR', 'Median RR', 'Label']
dataset = pd.DataFrame(data, columns=columns)

if dataset.empty:
    print("No valid data to process.")
else:
    # Plot feature distributions
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(columns[:-1]):
        plt.subplot(2, 2, i + 1)
        sns.histplot(dataset[column], kde=True)
        plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.show()

    # Plot label distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Label', data=dataset)
    plt.title('Label Distribution')
    plt.savefig('label_distribution.png')
    plt.show()

    # Trains a Logistic Regression model with regularization
    X = dataset[['SDNN', 'RMSSD', 'Mean RR', 'Median RR']]
    y = dataset['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluates accuracy and additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    plt.show()

    # Assuming patient IDs are stored
    patient_ids = X_test.index  # Use dataset index or extract IDs from filenames
    results = pd.DataFrame({'Patient_ID': patient_ids, 'Actual': y_test, 'Predicted': y_pred})

    try:
        # Save predictions to an Excel file
        results.to_excel('predictions.xlsx', index=False)
        print("Results saved to 'predictions.xlsx'.")
    except PermissionError:
        print("Permission denied: Unable to save 'predictions.xlsx'. Please close the file if it is open and try again.")

    # Print predictions for all patients
    print("Predictions for all patients:")
    print(results)

    try:
        # Save the full dataset with features and labels to a CSV file
        dataset.to_csv('extracted_features.csv', index=False)
        print("Extracted features saved to 'extracted_features.csv'.")
    except PermissionError:
        print("Permission denied: Unable to save 'extracted_features.csv'. Please close the file if it is open and try again.")