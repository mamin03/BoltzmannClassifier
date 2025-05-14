# BoltzmannClassifier

**A thermodynamic-inspired, interpretable classifier for supervised learning tasks.**

This repository contains the implementation of the **Boltzmann Classifier**, a novel machine learning algorithm that uses energy-based reasoning derived from the Boltzmann distribution. It offers interpretable, probabilistic classification and is compatible with the scikit-learn API.

---

## 🧠 Motivation

Inspired by principles from statistical mechanics, this classifier assigns probabilities to classes based on their "energy," computed as the L1 distance between an input vector and the class centroid. The result is a model that is:

- Simple and interpretable
- Thermodynamically grounded
- Fast and easy to implement
- Compatible with `scikit-learn` pipelines and tools

---

## 📦 Features

- Scikit-learn-style classifier (`fit`, `predict`, `predict_proba`)
- Uses **Boltzmann distribution** to compute class probabilities
- Customizable temperature (`T`) and scaling factor (`k`)
- Compatible with pipelines, grid search, and CV
- Fully tested on the Breast Cancer Wisconsin dataset

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mamin03/BoltzmannClassifier.git
cd boltzmann-classifier

### 2.  Install dependencies
pip install numpy scikit-learn

### 3. Run Example
python BoltzmannClassifier.py

On the Breast Cancer Wisconsin dataset, the Boltzmann Classifier achieves:

Accuracy: 95%
Precision (Malignant): 0.95
Recall (Malignant): 0.98
F1-score (Average): 0.96

License

This project is licensed under the MIT License. See the LICENSE file for details.


