"""
CSE 472 Assignment 2: Decision Trees, Random Forests, and Extra Trees

This module implements tree-based classification algorithms from scratch:
- Decision Tree
- Random Forest
- Extra Trees (Extremely Randomized Trees)

Author: [Your Name]
"""

import numpy as np
from collections import Counter
from typing import Optional, Tuple, List, Union
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# METRICS FUNCTIONS
# ============================================================================

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy."""
    return np.mean(y_true == y_pred)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """
    Calculate F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'macro' or 'weighted' averaging
    """
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores = []

    for label in unique_labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)

    if average == 'macro':
        return np.mean(f1_scores)
    elif average == 'weighted':
        weights = np.array([np.sum(y_true == label) for label in unique_labels])
        return np.sum(f1_scores * weights) / np.sum(weights)
    else:
        raise ValueError(f"Unknown average: {average}")


def auroc_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate Area Under ROC Curve (AUROC) using One-vs-Rest strategy.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities (n_samples, n_classes)
    """
    unique_labels = np.unique(y_true)
    n_classes = len(unique_labels)
    auroc_scores = []

    for i, label in enumerate(unique_labels):
        # Binary case: current class vs rest
        y_binary = (y_true == label).astype(int)
        y_scores = y_proba[:, i]

        # Sort by scores in descending order
        indices = np.argsort(y_scores)[::-1]
        y_binary_sorted = y_binary[indices]

        # Calculate ROC points
        n_pos = np.sum(y_binary)
        n_neg = len(y_binary) - n_pos

        if n_pos == 0 or n_neg == 0:
            auroc_scores.append(0.5)
            continue

        # Calculate AUC using trapezoidal rule
        # Start from (0, 0) and move through sorted samples
        tpr = 0.0
        fpr = 0.0
        auc = 0.0
        prev_fpr = 0.0
        
        for j in range(len(y_binary_sorted)):
            if y_binary_sorted[j] == 1:
                # True positive: move up (increase TPR)
                tpr += 1.0 / n_pos
            else:
                # False positive: move right (increase FPR)
                # Add area under the curve: rectangle with width = change in FPR, height = current TPR
                prev_fpr = fpr
                fpr += 1.0 / n_neg
                auc += (fpr - prev_fpr) * tpr

        auroc_scores.append(auc)

    # Return macro average
    return np.mean(auroc_scores)


# ============================================================================
# CRITERIA FUNCTIONS
# ============================================================================

def gini_impurity(y: np.ndarray) -> float:
    """Calculate Gini impurity for a set of labels."""
    if len(y) == 0:
        return 0
    y_int = y.astype(int)
    if np.any(y_int < 0):
        raise ValueError("Labels must be non-negative integers")
    counts = np.bincount(y_int)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)


def entropy(y: np.ndarray) -> float:
    """Calculate entropy for a set of labels."""
    if len(y) == 0:
        return 0
    y_int = y.astype(int)
    if np.any(y_int < 0):
        raise ValueError("Labels must be non-negative integers")
    counts = np.bincount(y_int)
    probs = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))


# ============================================================================
# DECISION TREE IMPLEMENTATION
# ============================================================================

class DecisionTreeNode:
    """Node in a decision tree."""

    def __init__(self, feature_idx: int = None, threshold: float = None,
                 left: 'DecisionTreeNode' = None, right: 'DecisionTreeNode' = None,
                 value: Union[int, np.ndarray] = None, is_leaf: bool = False):
        self.feature_idx = feature_idx      # Feature index to split on
        self.threshold = threshold          # Threshold value for split
        self.left = left                    # Left child (<= threshold)
        self.right = right                  # Right child (> threshold)
        self.value = value                  # Class label (for leaf nodes)
        self.is_leaf = is_leaf              # Whether this is a leaf node


class DecisionTreeClassifier:
    """
    Decision Tree Classifier from scratch.

    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of the tree. None means unlimited.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    criterion : str, default='gini'
        The function to measure the quality of a split ('gini' or 'entropy').
    random_state : int, default=None
        Random seed for reproducibility.
    """

    def __init__(self, max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 criterion: str = 'gini',
                 random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state
        self.root = None
        self.n_classes = None
        self.n_features = None

        if random_state is not None:
            np.random.seed(random_state)

    def _get_criterion_func(self):
        """Get the criterion function."""
        if self.criterion == 'gini':
            return gini_impurity
        elif self.criterion == 'entropy':
            return entropy
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _find_best_split(self, X: np.ndarray, y: np.ndarray,
                         feature_indices: List[int] = None) -> Tuple[int, float]:
        """
        Find the best split for a node.

        Returns:
            best_feature_idx, best_threshold
        """
        criterion_func = self._get_criterion_func()
        n_samples, n_features = X.shape

        if feature_indices is None:
            feature_indices = list(range(n_features))

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        parent_impurity = criterion_func(y)

        for feat_idx in feature_indices:
            feature_values = X[:, feat_idx]

            # Get unique thresholds
            unique_values = np.unique(feature_values)

            if len(unique_values) <= 1:
                continue

            # Consider midpoints between adjacent values as thresholds
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                # Weighted impurity of children
                left_impurity = criterion_func(y[left_mask])
                right_impurity = criterion_func(y[right_mask])

                weighted_impurity = (n_left / n_samples) * left_impurity + \
                                    (n_right / n_samples) * right_impurity

                # Information gain
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray,
                    depth: int = 0) -> DecisionTreeNode:
        """Recursively build the decision tree."""
        n_samples = X.shape[0]

        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            leaf_value = self._get_leaf_value(y)
            return DecisionTreeNode(value=leaf_value, is_leaf=True)

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)

        # No valid split found
        if best_feature is None:
            leaf_value = self._get_leaf_value(y)
            return DecisionTreeNode(value=leaf_value, is_leaf=True)

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursively build subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            is_leaf=False
        )

    def _get_leaf_value(self, y: np.ndarray) -> int:
        """Get the majority class for a leaf node."""
        if len(y) == 0:
            return 0
        counts = np.bincount(y.astype(int))
        return int(np.argmax(counts))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """Fit the decision tree to training data."""
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]

        # Reset random state for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.root = self._build_tree(X, y)
        return self

    def _predict_single(self, x: np.ndarray, node: DecisionTreeNode) -> int:
        """Predict class for a single sample."""
        if node.is_leaf:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        X = np.asarray(X)
        return np.array([self._predict_single(x, self.root) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.

        Returns:
            Array of shape (n_samples, n_classes) with probabilities.
        """
        X = np.asarray(X)
        proba = []

        for x in X:
            node = self.root
            while not node.is_leaf:
                if x[node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            # One-hot encode the leaf value
            p = np.zeros(self.n_classes)
            p[node.value] = 1.0
            proba.append(p)

        return np.array(proba)


# ============================================================================
# RANDOM FOREST IMPLEMENTATION
# ============================================================================

class _RandomForestTree(DecisionTreeClassifier):
    """Special decision tree for Random Forest with feature randomization."""
    
    def __init__(self, max_depth, min_samples_split, criterion, random_state, n_features_split, rng):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split,
                         criterion=criterion, random_state=random_state)
        self.n_features_split = n_features_split
        self._rng = rng
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray,
                         feature_indices: List[int] = None) -> Tuple[int, float]:
        """Override to use random feature subset."""
        criterion_func = self._get_criterion_func()
        n_samples, n_total_features = X.shape
        
        # Random feature selection
        feature_indices = self._rng.choice(n_total_features, 
                                          size=min(self.n_features_split, n_total_features), 
                                          replace=False)

        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        parent_impurity = criterion_func(y)

        for feat_idx in feature_indices:
            feature_values = X[:, feat_idx]
            unique_values = np.unique(feature_values)

            if len(unique_values) <= 1:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                left_impurity = criterion_func(y[left_mask])
                right_impurity = criterion_func(y[right_mask])

                weighted_impurity = (n_left / n_samples) * left_impurity + \
                                    (n_right / n_samples) * right_impurity

                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = threshold

        return best_feature, best_threshold


class RandomForestClassifier:
    """
    Random Forest Classifier from scratch.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=None
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    max_features : int, float, str, default='sqrt'
        Number of features to consider for best split.
    criterion : str, default='gini'
        Split quality criterion ('gini' or 'entropy').
    random_state : int, default=None
        Random seed for reproducibility.
    """

    def __init__(self, n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 max_features: Union[str, int, float] = 'sqrt',
                 criterion: str = 'gini',
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.trees = []
        self.n_classes = None

        # Setup random state
        self._rng = np.random.RandomState(random_state)

    def _get_n_features(self, n_features: int) -> int:
        """Get number of features to consider for split."""
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                return int(np.log2(n_features))
            else:
                raise ValueError(f"Unknown max_features: {self.max_features}")
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features  # Use all features

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create a bootstrap sample."""
        n_samples = X.shape[0]
        indices = self._rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """Fit the random forest to training data."""
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_classes = len(np.unique(y))
        n_features = X.shape[1]
        n_features_split = self._get_n_features(n_features)

        self.trees = []

        for i in range(self.n_estimators):
            # Bootstrap sample
            X_boot, y_boot = self._bootstrap_sample(X, y)

            # Create specialized tree with random feature selection
            tree_rng = np.random.RandomState(self.random_state + i if self.random_state is not None else None)
            tree = _RandomForestTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                random_state=self.random_state + i if self.random_state is not None else None,
                n_features_split=n_features_split,
                rng=tree_rng
            )

            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self



    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using majority voting."""
        X = np.asarray(X)

        # Collect predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Majority voting
        result = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            result.append(int(np.bincount(votes.astype(int)).argmax()))

        return np.array(result)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities by averaging tree predictions."""
        X = np.asarray(X)

        # Collect probability predictions from all trees
        probas = np.array([tree.predict_proba(X) for tree in self.trees])

        # Average probabilities
        return np.mean(probas, axis=0)


# ============================================================================
# EXTRA TREES IMPLEMENTATION
# ============================================================================

class ExtraTreesClassifier:
    """
    Extra Trees (Extremely Randomized Trees) Classifier from scratch.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=None
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    max_features : int, float, str, default='sqrt'
        Number of features to consider for best split.
    criterion : str, default='gini'
        Split quality criterion ('gini' or 'entropy').
    random_state : int, default=None
        Random seed for reproducibility.
    """

    def __init__(self, n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 max_features: Union[str, int, float] = 'sqrt',
                 criterion: str = 'gini',
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.trees = []
        self.n_classes = None

        self._rng = np.random.RandomState(random_state)

    def _get_n_features(self, n_features: int) -> int:
        """Get number of features to consider for split."""
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                return int(np.log2(n_features))
            else:
                raise ValueError(f"Unknown max_features: {self.max_features}")
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features

    def _extra_randomized_split(self, X: np.ndarray, y: np.ndarray,
                                 n_features: int, criterion_func) -> Tuple[int, float]:
        """
        Find split using Extra Trees strategy:
        - Random feature selection
        - Random threshold selection (no optimization)
        """
        n_samples, n_total_features = X.shape
        feature_indices = self._rng.choice(n_total_features, size=min(n_features, n_total_features), replace=False)

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        parent_impurity = criterion_func(y)

        for feat_idx in feature_indices:
            feature_values = X[:, feat_idx]
            min_val, max_val = feature_values.min(), feature_values.max()

            if min_val == max_val:
                continue

            # Random threshold - Extra Trees key feature!
            threshold = self._rng.uniform(min_val, max_val)

            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)

            left_impurity = criterion_func(y[left_mask])
            right_impurity = criterion_func(y[right_mask])

            weighted_impurity = (n_left / n_samples) * left_impurity + \
                                (n_right / n_samples) * right_impurity

            gain = parent_impurity - weighted_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feat_idx
                best_threshold = threshold

        return best_feature, best_threshold

    def _build_extra_tree(self, X: np.ndarray, y: np.ndarray,
                          depth: int, n_features_split: int, criterion_func) -> DecisionTreeNode:
        """Build a single Extra Tree."""
        n_samples = X.shape[0]

        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            leaf_value = self._get_leaf_value(y)
            return DecisionTreeNode(value=leaf_value, is_leaf=True)

        # Find split using Extra Trees strategy
        best_feature, best_threshold = self._extra_randomized_split(
            X, y, n_features_split, criterion_func
        )

        if best_feature is None:
            leaf_value = self._get_leaf_value(y)
            return DecisionTreeNode(value=leaf_value, is_leaf=True)

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_child = self._build_extra_tree(X[left_mask], y[left_mask], depth + 1, n_features_split, criterion_func)
        right_child = self._build_extra_tree(X[right_mask], y[right_mask], depth + 1, n_features_split, criterion_func)

        return DecisionTreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            is_leaf=False
        )

    def _get_leaf_value(self, y: np.ndarray) -> int:
        """Get the majority class for a leaf node."""
        if len(y) == 0:
            return 0
        counts = np.bincount(y.astype(int))
        return int(np.argmax(counts))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ExtraTreesClassifier':
        """Fit the Extra Trees classifier to training data."""
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_classes = len(np.unique(y))
        n_features = X.shape[1]
        n_features_split = self._get_n_features(n_features)

        criterion_func = gini_impurity if self.criterion == 'gini' else entropy

        self.trees = []

        for i in range(self.n_estimators):
            # Update RNG state for diversity across trees
            if self.random_state is not None:
                self._rng = np.random.RandomState(self.random_state + i)
            
            # Use entire dataset (no bootstrapping in Extra Trees by default)
            tree = self._build_extra_tree(X, y, depth=0, n_features_split=n_features_split, criterion_func=criterion_func)
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using majority voting."""
        X = np.asarray(X)
        predictions = []

        for x in X:
            votes = []
            for tree in self.trees:
                node = tree
                while not node.is_leaf:
                    if x[node.feature_idx] <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
                votes.append(node.value)

            predictions.append(int(np.bincount(votes).argmax()))

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities by averaging tree predictions."""
        X = np.asarray(X)
        probas = []

        for x in X:
            votes = []
            for tree in self.trees:
                node = tree
                while not node.is_leaf:
                    if x[node.feature_idx] <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
                votes.append(node.value)

            # Convert votes to probabilities
            p = np.zeros(self.n_classes)
            for v in votes:
                p[v] += 1
            p = p / len(votes)
            probas.append(p)

        return np.array(probas)


# ============================================================================
# DATASET STATISTICS FUNCTION
# ============================================================================

def print_dataset_statistics(dataset_name: str, X: np.ndarray, y: np.ndarray):
    """
    Print comprehensive statistics about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        X: Feature matrix
        y: Target labels
    """
    n_samples, n_features = X.shape
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)
    
    print(f"\n{'='*80}")
    print(f"Dataset Statistics: {dataset_name}")
    print(f"{'='*80}")
    print(f"\nBasic Information:")
    print(f"  - Number of samples: {n_samples}")
    print(f"  - Number of features: {n_features}")
    print(f"  - Number of classes: {n_classes}")
    
    print(f"\nClass Distribution:")
    for cls, count in zip(unique_classes, class_counts):
        percentage = (count / n_samples) * 100
        print(f"  - Class {int(cls)}: {count} samples ({percentage:.2f}%)")
    
    print(f"\nFeature Statistics:")
    print(f"  - Feature means: {np.mean(X, axis=0)}")
    print(f"  - Feature std devs: {np.std(X, axis=0)}")
    print(f"  - Feature mins: {np.min(X, axis=0)}")
    print(f"  - Feature maxs: {np.max(X, axis=0)}")
    
    # Class balance check
    balance_ratio = np.min(class_counts) / np.max(class_counts)
    print(f"\nClass Balance:")
    print(f"  - Balance ratio (min/max): {balance_ratio:.3f}")
    if balance_ratio < 0.5:
        print(f"  - WARNING: Dataset is imbalanced!")
    else:
        print(f"  - Dataset is reasonably balanced")
    
    print(f"{'='*80}\n")


# ============================================================================
# EVALUATION AND COMPARISON FUNCTIONS
# ============================================================================

def evaluate_model(model, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a model on train and test sets.

    Returns:
        Dictionary with accuracy, f1, and auroc for train and test sets.
    """
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Get probabilities for AUROC
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)
    else:
        # Fallback for sklearn models that always have predict_proba
        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)

    results = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'train_auroc': auroc_score(y_train, y_train_proba),
        'test_auroc': auroc_score(y_test, y_test_proba),
    }

    return results


def print_results_table(dataset_name: str, custom_results: dict, sklearn_results: dict):
    """Print comparison results in a formatted table."""
    print(f"\n{'='*80}")
    print(f"Results for {dataset_name} Dataset")
    print(f"{'='*80}")
    print(f"\nCustom Implementations:")
    print(f"{'-'*80}")
    print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Train F1':<12} {'Test F1':<12} {'Test AUROC':<12}")
    print(f"{'-'*80}")

    for model_name, results in custom_results.items():
        print(f"{model_name:<20} {results['train_accuracy']:<12.4f} {results['test_accuracy']:<12.4f} "
              f"{results['train_f1']:<12.4f} {results['test_f1']:<12.4f} {results['test_auroc']:<12.4f}")

    print(f"\nscikit-learn Implementations:")
    print(f"{'-'*80}")
    print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Train F1':<12} {'Test F1':<12} {'Test AUROC':<12}")
    print(f"{'-'*80}")

    for model_name, results in sklearn_results.items():
        print(f"{model_name:<20} {results['train_accuracy']:<12.4f} {results['test_accuracy']:<12.4f} "
              f"{results['train_f1']:<12.4f} {results['test_f1']:<12.4f} {results['test_auroc']:<12.4f}")

    print(f"{'='*80}\n")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """Run the complete comparison experiment."""
    from sklearn.datasets import load_iris, load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
    from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
    from sklearn.ensemble import ExtraTreesClassifier as SklearnExtraTrees

    # Set random seed for reproducibility
    RANDOM_STATE = 42

    print("\n" + "="*80)
    print("CSE 472 Assignment 2: Decision Trees, Random Forests, and Extra Trees")
    print("="*80)
    print("\nRunning experiments with random_state = 42...")

    # Common hyperparameters
    hyperparams = {
        'max_depth': 10,
        'min_samples_split': 2,
        'n_estimators': 50,
        'max_features': 'sqrt',
        'random_state': RANDOM_STATE
    }

    # =========================================================================
    # Iris Dataset
    # =========================================================================
    print("\n[1/2] Loading Iris dataset...")
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    # Print dataset statistics
    print_dataset_statistics("Iris", X_iris, y_iris)

    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=RANDOM_STATE, stratify=y_iris
    )
    
    print(f"\nTrain/Test Split:")
    print(f"  - Training samples: {len(y_train)}")
    print(f"  - Test samples: {len(y_test)}")

    # Custom implementations
    print("  - Training Custom Decision Tree...")
    custom_dt_iris = evaluate_model(
        DecisionTreeClassifier(max_depth=hyperparams['max_depth'],
                               min_samples_split=hyperparams['min_samples_split'],
                               random_state=hyperparams['random_state']),
        X_train, y_train, X_test, y_test
    )

    print("  - Training Custom Random Forest...")
    custom_rf_iris = evaluate_model(
        RandomForestClassifier(n_estimators=hyperparams['n_estimators'],
                               max_depth=hyperparams['max_depth'],
                               min_samples_split=hyperparams['min_samples_split'],
                               max_features=hyperparams['max_features'],
                               random_state=hyperparams['random_state']),
        X_train, y_train, X_test, y_test
    )

    print("  - Training Custom Extra Trees...")
    custom_et_iris = evaluate_model(
        ExtraTreesClassifier(n_estimators=hyperparams['n_estimators'],
                             max_depth=hyperparams['max_depth'],
                             min_samples_split=hyperparams['min_samples_split'],
                             max_features=hyperparams['max_features'],
                             random_state=hyperparams['random_state']),
        X_train, y_train, X_test, y_test
    )

    # scikit-learn implementations
    print("  - Training sklearn Decision Tree...")
    sklearn_dt_iris = evaluate_model(
        SklearnDecisionTree(max_depth=hyperparams['max_depth'],
                            min_samples_split=hyperparams['min_samples_split'],
                            random_state=hyperparams['random_state']),
        X_train, y_train, X_test, y_test
    )

    print("  - Training sklearn Random Forest...")
    sklearn_rf_iris = evaluate_model(
        SklearnRandomForest(n_estimators=hyperparams['n_estimators'],
                            max_depth=hyperparams['max_depth'],
                            min_samples_split=hyperparams['min_samples_split'],
                            max_features=hyperparams['max_features'],
                            random_state=hyperparams['random_state']),
        X_train, y_train, X_test, y_test
    )

    print("  - Training sklearn Extra Trees...")
    sklearn_et_iris = evaluate_model(
        SklearnExtraTrees(n_estimators=hyperparams['n_estimators'],
                          max_depth=hyperparams['max_depth'],
                          min_samples_split=hyperparams['min_samples_split'],
                          max_features=hyperparams['max_features'],
                          random_state=hyperparams['random_state']),
        X_train, y_train, X_test, y_test
    )

    print_results_table(
        "Iris",
        {
            'Decision Tree': custom_dt_iris,
            'Random Forest': custom_rf_iris,
            'Extra Trees': custom_et_iris
        },
        {
            'Decision Tree': sklearn_dt_iris,
            'Random Forest': sklearn_rf_iris,
            'Extra Trees': sklearn_et_iris
        }
    )

    # =========================================================================
    # Wine Dataset
    # =========================================================================
    print("\n[2/2] Loading Wine dataset...")
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target
    
    # Print dataset statistics
    print_dataset_statistics("Wine", X_wine, y_wine)

    X_train, X_test, y_train, y_test = train_test_split(
        X_wine, y_wine, test_size=0.3, random_state=RANDOM_STATE, stratify=y_wine
    )
    
    print(f"\nTrain/Test Split:")
    print(f"  - Training samples: {len(y_train)}")
    print(f"  - Test samples: {len(y_test)}")

    # Custom implementations
    print("  - Training Custom Decision Tree...")
    custom_dt_wine = evaluate_model(
        DecisionTreeClassifier(max_depth=hyperparams['max_depth'],
                               min_samples_split=hyperparams['min_samples_split'],
                               random_state=hyperparams['random_state']),
        X_train, y_train, X_test, y_test
    )

    print("  - Training Custom Random Forest...")
    custom_rf_wine = evaluate_model(
        RandomForestClassifier(n_estimators=hyperparams['n_estimators'],
                               max_depth=hyperparams['max_depth'],
                               min_samples_split=hyperparams['min_samples_split'],
                               max_features=hyperparams['max_features'],
                               random_state=hyperparams['random_state']),
        X_train, y_train, X_test, y_test
    )

    print("  - Training Custom Extra Trees...")
    custom_et_wine = evaluate_model(
        ExtraTreesClassifier(n_estimators=hyperparams['n_estimators'],
                             max_depth=hyperparams['max_depth'],
                             min_samples_split=hyperparams['min_samples_split'],
                             max_features=hyperparams['max_features'],
                             random_state=hyperparams['random_state']),
        X_train, y_train, X_test, y_test
    )

    # scikit-learn implementations
    print("  - Training sklearn Decision Tree...")
    sklearn_dt_wine = evaluate_model(
        SklearnDecisionTree(max_depth=hyperparams['max_depth'],
                            min_samples_split=hyperparams['min_samples_split'],
                            random_state=hyperparams['random_state']),
        X_train, y_train, X_test, y_test
    )

    print("  - Training sklearn Random Forest...")
    sklearn_rf_wine = evaluate_model(
        SklearnRandomForest(n_estimators=hyperparams['n_estimators'],
                            max_depth=hyperparams['max_depth'],
                            min_samples_split=hyperparams['min_samples_split'],
                            max_features=hyperparams['max_features'],
                            random_state=hyperparams['random_state']),
        X_train, y_train, X_test, y_test
    )

    print("  - Training sklearn Extra Trees...")
    sklearn_et_wine = evaluate_model(
        SklearnExtraTrees(n_estimators=hyperparams['n_estimators'],
                          max_depth=hyperparams['max_depth'],
                          min_samples_split=hyperparams['min_samples_split'],
                          max_features=hyperparams['max_features'],
                          random_state=hyperparams['random_state']),
        X_train, y_train, X_test, y_test
    )

    print_results_table(
        "Wine",
        {
            'Decision Tree': custom_dt_wine,
            'Random Forest': custom_rf_wine,
            'Extra Trees': custom_et_wine
        },
        {
            'Decision Tree': sklearn_dt_wine,
            'Random Forest': sklearn_rf_wine,
            'Extra Trees': sklearn_et_wine
        }
    )

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY: Custom vs scikit-learn Performance Comparison")
    print("="*80)
    print("\nKey Observations:")
    print("1. Ensemble methods (RF, Extra Trees) generally outperform single Decision Trees")
    print("2. Extra Trees often achieve similar accuracy with more randomness (faster training)")
    print("3. Custom implementations show competitive results with sklearn")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
