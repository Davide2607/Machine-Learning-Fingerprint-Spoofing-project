# Fingerprint Spoofing Detection

**Author:** Davide Ravidà  
**Course:** Machine Learning for Pattern Recognition  
**Program:** Data Analysis and Artificial Intelligence (2024/2025)

---

## 📖 Project Overview

This project addresses a **binary classification problem** focused on fingerprint spoofing detection.

The objective is to distinguish between:

- **Genuine fingerprints** (label = 1)
- **Spoofed / counterfeit fingerprints** (label = 0)

The dataset consists of feature vectors extracted from fingerprint images.  
Each sample is represented by a **6-dimensional feature vector**, summarizing high-level characteristics of the original image.

---

## 🎯 Objective

Develop and evaluate machine learning models capable of accurately detecting spoofed fingerprints.

---

## 🧠 Dataset

- Binary classification task
- 6-dimensional feature vectors
- Labels:
  - 1 → Genuine
  - 0 → Fake

---
## Repository contents

Main files:

- `main.py`: experiment “driver” script. Most experiments are enabled/disabled by commenting/uncommenting code blocks.
- `DataAnalysis.py`: exploratory analysis and plotting utilities.
- `PCA.py`, `LDA.py`: dimensionality reduction (PCA) and discriminant projection (LDA).
- `Gaussian_distribution.py`: Gaussian/density helper functions.
- `MVG_model.py`: Gaussian classifiers (MVG / Naive Bayes / Tied Covariance) + score computation (LLR).
- `Logistic_regression.py`: logistic regression utilities used by the experiments (including calibration-related routines where used).
- `SVM_linear_kernel.py`: SVM models (linear and kernel-based where implemented).
- `GMM.py`: Gaussian Mixture Models (LBG training + covariance variants).
- `Evaluations.py`: evaluation metrics and plots (minDCF/actDCF, Bayes error plots, etc.).
- `Relazione_S324381.pdf`: project report (detailed discussion of methods and results).

Data files currently present in the repository root:
- `trainData.txt`
- `evalData.txt`
---

## 📈 Evaluation

Models are evaluated using appropriate classification metrics such as:

- Accuracy
- Precision / Recall
- ROC Curve
- Detection Error Tradeoff (DET)
- Confusion Matrix

---

## 🛠 Technologies Used

- Python
- NumPy
- Matplotlib

---

## 📌 Notes

This project was developed for academic purposes.

