"""
Unified classification module.

Classifiers: Logistic Regression, XGBoost, Shallow Neural Network.
Input types: difference vectors, concatenation vectors.
Tasks: pairwise binary, multiclass.
"""

import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

logger = logging.getLogger(__name__)

InputType = Literal["difference", "concatenation"]
PairList = List[Tuple[str, str]]


# ---- Feature construction ----


def symmetrize_features(X: np.ndarray, base_input_type: str) -> np.ndarray:
    """Double feature matrix by adding symmetric counterparts.

    For difference: adds -X (negated vectors).
    For concatenation: adds [e2||e1] (swapped halves).
    """
    if base_input_type == "difference":
        return np.vstack([X, -X])
    elif base_input_type == "concatenation":
        d = X.shape[1] // 2
        X_swapped = np.hstack([X[:, d:], X[:, :d]])
        return np.vstack([X, X_swapped])
    else:
        raise ValueError(f"Unknown base_input_type for symmetrization: {base_input_type}")


def build_features(
    emb_pairs: List[Tuple[np.ndarray, np.ndarray]],
    input_type: InputType,
) -> np.ndarray:
    """Build feature matrix from embedding pairs.

    Args:
        emb_pairs: List of (emb1, emb2) arrays.
        input_type: "difference" (emb2 - emb1) or "concatenation" (emb1 || emb2).

    Returns:
        Feature matrix of shape (n_pairs, feature_dim).
    """
    features = []
    for e1, e2 in emb_pairs:
        if input_type == "difference":
            features.append(e2 - e1)
        elif input_type == "concatenation":
            features.append(np.concatenate([e1, e2]))
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
    return np.array(features) if features else np.empty((0, 0))


# ---- Class balancing ----

def balance_classes(
    X_list: List[np.ndarray],
    class_names: List[str],
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Undersample all classes to the size of the smallest class.

    Returns:
        X: Balanced feature matrix.
        y: Balanced labels (integer-encoded).
    """
    min_size = min(len(X) for X in X_list)
    X_balanced, y_balanced = [], []

    for i, (X, name) in enumerate(zip(X_list, class_names)):
        if len(X) > min_size:
            X_sampled = resample(X, n_samples=min_size, random_state=random_state, replace=False)
        else:
            X_sampled = X
        X_balanced.append(X_sampled)
        y_balanced.append(np.full(len(X_sampled), i))
        logger.info("  %s: %d -> %d samples", name, len(X), len(X_sampled))

    return np.vstack(X_balanced), np.hstack(y_balanced)


# ---- Word-aware splitting ----

def word_aware_split(
    X_list: List[np.ndarray],
    pairs_list: List[PairList],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Split pairs into train/test ensuring no word appears in both splits.

    Collects all unique words across all classes, splits that vocabulary
    into train-words and test-words, then assigns each pair to train or
    test only if both its words belong to the same split. Pairs that
    straddle the boundary are dropped.

    Args:
        X_list: Feature arrays, one per class (aligned with pairs_list).
        pairs_list: Word-pair lists, one per class.
        test_size: Fraction of the vocabulary to hold out for testing.
        random_state: Random seed.

    Returns:
        X_train_list: Feature arrays for training, one per class.
        X_test_list: Feature arrays for testing, one per class.
    """
    all_words = sorted({w for pairs in pairs_list for w1, w2 in pairs for w in (w1, w2)})
    train_words_list, test_words_list = train_test_split(
        all_words, test_size=test_size, random_state=random_state,
    )
    train_words = set(train_words_list)
    test_words = set(test_words_list)

    n_dropped = 0
    X_train_list, X_test_list = [], []
    for X, pairs in zip(X_list, pairs_list):
        train_idx = [i for i, (w1, w2) in enumerate(pairs) if w1 in train_words and w2 in train_words]
        test_idx = [i for i, (w1, w2) in enumerate(pairs) if w1 in test_words and w2 in test_words]
        n_dropped += len(pairs) - len(train_idx) - len(test_idx)
        X_train_list.append(X[train_idx])
        X_test_list.append(X[test_idx])

    logger.info(
        "Word-aware split: %d train-words, %d test-words, %d bridge pairs dropped",
        len(train_words), len(test_words), n_dropped,
    )
    return X_train_list, X_test_list


# ---- Classifiers ----

def train_logistic(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    class_names: List[str],
    max_iter: int = 1000,
    random_state: int = 42,
) -> Dict:
    """Train and evaluate logistic regression."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=max_iter, random_state=random_state, n_jobs=-1)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)

    return _evaluate(y_test, y_pred, class_names, "logistic")


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    class_names: List[str],
    random_state: int = 42,
) -> Dict:
    """Train and evaluate XGBoost classifier."""
    from xgboost import XGBClassifier

    n_classes = len(class_names)
    params = dict(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=1,
    )
    if n_classes > 2:
        params["objective"] = "multi:softmax"
        params["num_class"] = n_classes
    else:
        params["objective"] = "binary:logistic"
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return _evaluate(y_test, y_pred, class_names, "xgboost")


def train_shallow_nn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    class_names: List[str],
    hidden_dim: int = 64,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 10,
) -> Dict:
    """Train and evaluate a shallow neural network (1 hidden layer)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    n_classes = len(class_names)
    input_dim = X_train.shape[1]

    # Standardize
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train).astype(np.float32)
    X_te = scaler.transform(X_test).astype(np.float32)

    # Build model
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, n_classes),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Data loaders
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_train.astype(np.int64)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Training loop with early stopping
    best_loss = float("inf")
    wait = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te))
        y_pred = logits.argmax(dim=1).numpy()

    return _evaluate(y_test, y_pred, class_names, "shallow_nn")


# ---- Evaluation ----

def _evaluate(
    y_true: np.ndarray, y_pred: np.ndarray,
    class_names: List[str], classifier_name: str,
) -> Dict:
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    logger.info("%s accuracy: %.4f", classifier_name, acc)
    return {
        "classifier": classifier_name,
        "accuracy": float(acc),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "class_names": class_names,
    }


# ---- High-level runner ----

def run_classification(
    X_list: List[np.ndarray],
    class_names: List[str],
    pairs_list: Optional[List[PairList]] = None,
    classifiers: List[str] = ("logistic", "xgboost", "shallow_nn"),
    test_size: float = 0.2,
    random_state: int = 42,
    symmetrize: bool = False,
    base_input_type: Optional[str] = None,
    **kwargs,
) -> List[Dict]:
    """Run multiple classifiers on balanced data.

    Args:
        X_list: Feature arrays, one per class.
        class_names: Names for each class.
        pairs_list: Word-pair lists, one per class. When provided, uses
            word-aware splitting so no base word appears in both train and
            test. Balancing is applied after splitting. When omitted, falls
            back to random pair-level splitting with balancing first.
        classifiers: Which classifiers to run.
        test_size: Fraction for test split.
        random_state: Random seed.
        symmetrize: If True, augment train and test sets with symmetric
            counterparts after splitting and balancing.
        base_input_type: Required when symmetrize is True. One of
            "difference" or "concatenation".
        **kwargs: Extra params (max_iter, hidden_dim, etc.)

    Returns:
        List of result dicts, one per classifier.
    """
    if symmetrize and base_input_type is None:
        raise ValueError("base_input_type is required when symmetrize=True")

    if pairs_list is not None:
        X_train_list, X_test_list = word_aware_split(
            X_list, pairs_list, test_size=test_size, random_state=random_state,
        )
        if any(len(X) == 0 for X in X_train_list) or any(len(X) == 0 for X in X_test_list):
            empty_train = [class_names[i] for i, X in enumerate(X_train_list) if len(X) == 0]
            empty_test = [class_names[i] for i, X in enumerate(X_test_list) if len(X) == 0]
            logger.warning(
                "Word-aware split produced empty class(es) — skipping task. "
                "Empty in train: %s. Empty in test: %s.",
                empty_train, empty_test,
            )
            return []
        X_train, y_train = balance_classes(X_train_list, class_names, random_state)
        X_test, y_test = balance_classes(X_test_list, class_names, random_state)
    else:
        X, y = balance_classes(X_list, class_names, random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )

    if symmetrize:
        X_train = symmetrize_features(X_train, base_input_type)
        y_train = np.tile(y_train, 2)
        X_test = symmetrize_features(X_test, base_input_type)
        y_test = np.tile(y_test, 2)

    results = []
    for clf_name in classifiers:
        logger.info("Training %s on %s...", clf_name, " vs ".join(class_names))
        if clf_name == "logistic":
            r = train_logistic(
                X_train, y_train, X_test, y_test, class_names,
                max_iter=kwargs.get("max_iter", 1000),
                random_state=random_state,
            )
        elif clf_name == "xgboost":
            r = train_xgboost(
                X_train, y_train, X_test, y_test, class_names,
                random_state=random_state,
            )
        elif clf_name == "shallow_nn":
            r = train_shallow_nn(
                X_train, y_train, X_test, y_test, class_names,
                hidden_dim=kwargs.get("hidden_dim", 64),
                epochs=kwargs.get("epochs", 100),
                batch_size=kwargs.get("batch_size", 64),
                learning_rate=kwargs.get("learning_rate", 1e-3),
                patience=kwargs.get("patience", 10),
            )
        else:
            logger.warning("Unknown classifier: %s", clf_name)
            continue
        results.append(r)

    return results
