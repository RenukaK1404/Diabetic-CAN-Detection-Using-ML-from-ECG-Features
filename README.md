# Diabetic-CAN-Detection-Using-ML-from-ECG-Features
A machine learning-based approach for the early detection of Cardiovascular Autonomic Neuropathy (CAN) in type 2 diabetes patients using ECG-derived heart rate variability features. Implements and compares Logistic Regression, Random Forest, and XGBoost models for non-invasive clinical screening.

This repository presents a research-based machine learning framework for the **early, accurate, and non-invasive diagnosis of Cardiovascular Autonomic Neuropathy (CAN)** in patients with type 2 diabetes. The approach utilizes **electrocardiogram (ECG)-derived Heart Rate Variability (HRV) features** and compares the performance of multiple classifiers—**Logistic Regression, Random Forest, and XGBoost**—to identify autonomic dysfunction with clinical relevance.

## Project Motivation

**Cardiovascular Autonomic Neuropathy (CAN)** is one of the most severe complications of diabetes, often underdiagnosed due to limitations in current clinical methods. Traditional tests such as Ewing’s battery, tilt-table testing, and MIBG imaging are time-consuming, invasive, and require specialized infrastructure.

This project aims to:

- Propose a **non-invasive, cost-effective, and ML-integrated** diagnostic approach.
- Leverage **ECG signal processing and HRV feature engineering**.
- Compare and validate multiple machine learning algorithms for CAN detection.
- Contribute toward practical clinical deployment and **digital health transformation**.


## 🔍 Problem Statement

To develop a machine learning model that can reliably detect **Cardiovascular Autonomic Neuropathy (CAN)** from **preprocessed ECG signals** using statistically significant **HRV features** derived from time-domain analysis.


## Workflow Overview

### 1️⃣ Dataset Acquisition
- Sourced from an open-access ECG database.
- Contains data from **25 CAN-diagnosed** and **25 normal individuals**.
- ECGs were recorded using wearable sensors (e.g., Zephyr BioHarness 3).

### 2️⃣ Data Preprocessing
- **Signal Filtering:** Bandpass filter (0.5–40 Hz) to remove motion artifacts, noise, and baseline drift.
- **R-Peak Detection:** Using `NeuroKit2` for precise localization of R-R intervals.
- **Baseline Correction:** Ensures the ECG waveform is stable before feature extraction.

### 3️⃣ Feature Extraction (Time Domain HRV)
- **SDNN** – Standard deviation of normal-to-normal RR intervals.
- **RMSSD** – Root mean square of successive differences.
- **Mean RR** and **Median RR** – Overall heart rhythm trend metrics.

> These are physiological markers known to be sensitive to autonomic imbalance.

### 4️⃣ Feature Labeling
- **Clinical Thresholds** used to label:
  - `SDNN < 17.13 ms` or `RMSSD < 24.94 ms` → CAN-positive (1)
  - Otherwise → Normal (0)

### 5️⃣ Model Development
- **Training/Test Split:** 80:20 for stratified validation.
- Models Implemented:
  - `Logistic Regression` — Interpretable and simple.
  - `Random Forest` — Ensemble-based, handles feature interaction.
  - `XGBoost` — Gradient boosting with regularization.
- **Cross-validation and hyperparameter tuning** performed.

### 6️⃣ Evaluation Metrics
Each model is assessed using:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- Confusion Matrix
- ROC Curve and AUC

## 📊 Results Summary

| Model              | Accuracy | Precision | Recall | F1 Score | AUC  |
|-------------------|----------|-----------|--------|----------|------|
| Logistic Regression | 90.00%   | 83.33%    | 100%   | 90.91%   | 1.00 |
| Random Forest       | 90.00%   | 83.33%    | 100%   | 90.91%   | 1.00 |
| XGBoost             | 80.00%   | 71.43%    | 100%   | 83.33%   | 0.96 |

> Logistic Regression and Random Forest models provided **optimal balance** of sensitivity and interpretability.

## ⚙️ Steps to Set Up Locally

Follow the guide below to reproduce results on your workstation.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YourUsername/ML-ECG-CAN-Detection.git
cd ML-ECG-CAN-Detection
```

### 2️⃣ Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost neurokit2 scipy
```

### 3️⃣ Run the Pipeline

```bash
python src/preprocessing.py
python src/feature_extraction.py
```

**Train & evaluate models**
```bash
python src/model_training.py
```

### 4️⃣ Output Artefacts
- **`results/`** – ROC curves, confusion matrices, metric CSVs  
- **`models/`** – Trained classifiers (`.pkl`) ready for inference  
- **`data/processed/`** – Labeled HRV-feature tables  

---

## 💡 Future Directions
- Increase dataset size & diversity for stronger generalisation.  
- Integrate deep-learning architectures (CNN/LSTM) for raw-signal learning.  
- Extend to **severity grading** of CAN, not just binary detection.  
- Deploy as a **Streamlit or mobile app** for bedside/wearable use.  
- Fuse multimodal data (ECG + EDA + BPV) for holistic autonomic assessment.  

---

## 🛠️ Key Technologies
`Python 3.x` • `NumPy` • `Pandas` • `SciPy` • `scikit-learn` • `XGBoost` • `NeuroKit2` • `Matplotlib` • `Seaborn`

---
