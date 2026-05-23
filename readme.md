# CSE 472: Machine Learning Sessional

This repository contains my coursework for **CSE 472: Machine Learning Sessional**. The work is organized by assignment and practice session, with notebooks, scripts, reports, and supporting files kept close to the task they belong to.

The focus of the repository is implementation-oriented learning: building preprocessing pipelines, training neural models, implementing classical algorithms from scratch, and comparing those implementations against standard library baselines.

## Contents

| Directory | Work | Summary |
| --- | --- | --- |
| `Offline 1/` | Data preprocessing and FNN validation | Cleans and prepares a medical student dataset, applies feature engineering and scaling, studies correlations, and validates the pipeline with a PyTorch feed-forward neural network. |
| `Offline 2/` | Decision trees and ensembles | Implements Decision Tree, Random Forest, and Extra Trees classifiers from scratch, then compares them with scikit-learn on Iris and Wine datasets. |
| `Online_practice/` | CNN architecture practice | Contains PyTorch practice work for MNIST/CIFAR-style image classification, including compact convolutional models such as SqueezeNet-like and MobileNet-style architectures. |

## Assignment Notes

### Offline 1: Data Preprocessing

The first offline assignment works through a complete preprocessing pipeline for the medical students dataset. The notebook covers dataset inspection, missing-value handling, target construction, categorical-to-numeric conversion, feature scaling, correlation analysis, and a final validation step using a feed-forward neural network in PyTorch.

Key files:

- `Offline 1/2005100.ipynb`
- `Offline 1/July25_CSE472_Assignment1.pdf`
- `Offline 1/medical_students_dataset - medical_students_dataset.csv`

### Offline 2: Tree-Based Learning

The second offline assignment implements tree-based classifiers without relying on scikit-learn for the core algorithms. The custom implementation includes impurity measures, recursive tree construction, bootstrap aggregation, random feature selection, randomized thresholds, probability prediction, and evaluation metrics.

The experiments compare custom models with scikit-learn implementations using accuracy, macro F1 score, and one-vs-rest AUROC.

Key files:

- `Offline 2/decision_trees_assignment.py`
- `Offline 2/2005100.ipynb`
- `Offline 2/REPORT.md`
- `Offline 2/CSE_472_Assignment_2_DT.pdf`

### Online Practice: CNN Architectures

The online practice folder contains PyTorch exercises for image classification. The completed work includes a SqueezeNet-inspired MNIST classifier using Fire modules. The folder also includes reference and practice files for conventional CNNs, Network-in-Network style models, and a MobileNetV1-style CIFAR-10 template.

Key files:

- `Online_practice/2005100.py`
- `Online_practice/cnn.py`
- `Online_practice/Online-B1.py`
- `Online_practice/B1solve.ipynb`
- `Online_practice/1.ipynb`
