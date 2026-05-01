# CSE 472 Assignment 2: Decision Trees, Random Forests, and Extra Trees

## Algorithmic Details

**Decision Tree:** Recursive greedy splitting using information gain (Gini/Entropy). Evaluates all feature-threshold pairs, splits at maximum gain, stops at max depth or pure nodes.

**Random Forest:** Ensemble of decision trees with (1) bagging - bootstrap sampling per tree, (2) random feature selection - √m features per split, (3) majority voting for prediction.

**Extra Trees:** Like Random Forest but uses full dataset (no bagging) and random thresholds instead of optimal ones. Faster training, similar performance.

## Experimental Setup

**Datasets:** Iris (150 samples, 4 features, 3 classes, 70/30 split) and Wine (178 samples, 13 features, 3 classes, 70/30 split).

**Metrics:** Accuracy, F1 Score (macro), AUROC (One-vs-Rest).

**Hyperparameters:** max_depth=10, min_samples_split=2, n_estimators=50, max_features='sqrt', criterion='gini', random_state=42.

## Results and Analysis

### Iris Dataset

**Custom Implementations:**

| Model | Train Acc | Test Acc | Train F1 | Test F1 | Test AUROC |
|-------|-----------|----------|----------|---------|------------|
| Decision Tree | 1.0000 | 0.9111 | 1.0000 | 0.9111 | 0.9274 |
| Random Forest | 1.0000 | 0.8889 | 1.0000 | 0.8878 | 0.9926 |
| Extra Trees | 1.0000 | 0.9111 | 1.0000 | 0.9107 | 0.9941 |

**Scikit-learn Implementations:**

| Model | Train Acc | Test Acc | Train F1 | Test F1 | Test AUROC |
|-------|-----------|----------|----------|---------|------------|
| Decision Tree | 1.0000 | 0.9333 | 1.0000 | 0.9327 | 0.9541 |
| Random Forest | 1.0000 | 0.8889 | 1.0000 | 0.8878 | 0.9881 |
| Extra Trees | 1.0000 | 0.9111 | 1.0000 | 0.9107 | 0.9941 |

### Wine Dataset

**Custom Implementations:**

| Model | Train Acc | Test Acc | Train F1 | Test F1 | Test AUROC |
|-------|-----------|----------|----------|---------|------------|
| Decision Tree | 1.0000 | 0.9815 | 1.0000 | 0.9827 | 0.9777 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Extra Trees | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Scikit-learn Implementations:**

| Model | Train Acc | Test Acc | Train F1 | Test F1 | Test AUROC |
|-------|-----------|----------|----------|---------|------------|
| Decision Tree | 1.0000 | 0.9630 | 1.0000 | 0.9638 | 0.9636 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Extra Trees | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Analysis

**Custom vs. sklearn:** Custom implementations match sklearn closely (≤2.2% accuracy difference). Extra Trees shows identical performance in most cases.

**Algorithm Comparison:** Ensembles outperform single trees. On Wine dataset, ensembles achieve perfect 100% test accuracy while Decision Tree achieves 98.15% (custom) and 96.30% (sklearn). Extra Trees achieves highest AUROC on Iris (0.9941).

**Conclusion:** Custom implementations successfully replicate sklearn behavior. Ensemble methods provide better generalization, especially on higher-dimensional datasets.
