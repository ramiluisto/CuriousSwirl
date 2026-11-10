"""Projection-free analyses of antonym/synonym pair geometry.

Everything here works directly in the embedding space and is invariant to
the order of the rows, so none of it can pick up the row-order artefact that
affects seeded UMAP layouts (see :mod:`semsim.order_artefact`).

Two facts shape the design:

* **Sign symmetry.**  Antonymy and synonymy are symmetric relations, so the
  pair ``(a, b)`` and the pair ``(b, a)`` carry the same label while their
  difference vectors are ``D`` and ``-D``.  Any *odd* function of ``D`` --
  in particular every linear probe ``w . D + c`` -- therefore has the same
  expected value on both classes once pair order is arbitrary.  Whatever
  separates the classes has to be an *even* function of ``D``: its length,
  ``|D|`` coordinate-wise, or a quadratic form ``D^T A D``.
* **Lexical memorisation.**  The same word occurs in many pairs.  A random
  train/test split of pairs lets a classifier recognise words rather than
  relations (Levy et al., 2015), so every evaluation is reported for a
  word-disjoint ("lexical") split as well.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

PairList = List[Tuple[str, str]]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def lexical_split(pairs: PairList, test_size: float = 0.2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Word-disjoint split: no word occurs in both a training and a test pair.

    Each word is assigned to the test vocabulary with probability
    ``test_size``; a pair goes to train (test) when both of its words are
    training (test) words and is dropped otherwise.  The assignment is a
    deterministic function of the word and the seed, so different models
    (with different vocabularies) hold out the same words.

    Returns boolean masks ``(train, test)`` over ``pairs``.
    """
    import zlib

    words = {w for pair in pairs for w in pair}
    is_test_word = {w: zlib.crc32(f"{seed}:{w}".encode()) / 2 ** 32 < test_size for w in words}
    train = np.array([not is_test_word[a] and not is_test_word[b] for a, b in pairs])
    test = np.array([is_test_word[a] and is_test_word[b] for a, b in pairs])
    return train, test


# ---------------------------------------------------------------------------
# Baselines that never look at an embedding
# ---------------------------------------------------------------------------

def word_overlap_vote(pairs: PairList, y: Sequence[int], train: np.ndarray, test: np.ndarray) -> Dict[str, float]:
    """Classify a test pair from the training labels of pairs that share a word with it.

    No embedding is used: this measures how much of the task is solved by the
    pair graph of the dataset alone.  Test pairs without any shared word (or
    with tied votes) get the training majority class.

    Returns the accuracy, the fraction of test pairs that share a word with a
    training pair (``coverage``) and the accuracy on that covered subset
    (``None`` when no test pair is covered, as on a word-disjoint split).
    """
    y = np.asarray(y)
    votes = defaultdict(Counter)
    for (a, b), label, is_train in zip(pairs, y, train):
        if is_train:
            votes[a][label] += 1
            votes[b][label] += 1
    majority = Counter(y[train]).most_common(1)[0][0]
    predictions, covered = [], []
    for (a, b), is_test in zip(pairs, test):
        if not is_test:
            continue
        tally = votes.get(a, Counter()) + votes.get(b, Counter())
        ranked = tally.most_common(2)
        covered.append(bool(ranked))
        if not ranked or (len(ranked) == 2 and ranked[0][1] == ranked[1][1]):
            predictions.append(majority)
        else:
            predictions.append(ranked[0][0])
    predictions, covered, truth = np.array(predictions), np.array(covered), y[test]
    return {
        "accuracy": float((predictions == truth).mean()),
        "coverage": float(covered.mean()),
        "accuracy_covered": float((predictions[covered] == truth[covered]).mean()) if covered.any() else None,
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _logreg_accuracy(F_train, y_train, F_test, y_test, C: float = 0.01) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(scaler.transform(F_train), y_train)
    return float(clf.score(scaler.transform(F_test), y_test))


def probe_accuracies(E1: np.ndarray, E2: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
                     seed: int = 0) -> Dict[str, float]:
    """Accuracy of simple probes on features of a word pair.

    ``cosine``         logistic regression on the cosine similarity alone
    ``linear_diff``    logistic regression on ``D = E2 - E1`` (odd in ``D``)
    ``linear_diff_symmetrised`` the same after randomising pair order, which
                       enforces the sign symmetry exactly
    ``abs_diff``       logistic regression on ``|D|`` coordinate-wise (even)
    ``product``        logistic regression on ``E1 * E2`` coordinate-wise (even)
    """
    y = np.asarray(y)
    D = E2 - E1
    cosine = np.einsum("ij,ij->i", E1, E2) / (np.linalg.norm(E1, axis=1) * np.linalg.norm(E2, axis=1))
    flip = np.where(np.random.default_rng(seed).random(len(D)) < 0.5, -1.0, 1.0)[:, None]
    features = {
        "cosine": cosine[:, None],
        "linear_diff": D,
        "linear_diff_symmetrised": D * flip,
        "abs_diff": np.abs(D),
        "product": E1 * E2,
    }
    return {name: _logreg_accuracy(F[train], y[train], F[test], y[test]) for name, F in features.items()}


# ---------------------------------------------------------------------------
# Second-moment contrast: is there an "antonymy subspace"?
# ---------------------------------------------------------------------------

def second_moment_contrast(D: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
                           n_components: int = 300, shrinkage: float = 0.3,
                           ranks: Sequence[int] = (1, 2, 3, 5, 10, 20, 50, 100, 300)) -> Dict:
    """Contrast the second moments of antonym and synonym difference directions.

    With sign symmetry the first moments of both classes vanish, so the
    simplest statistic that can differ is the second-moment matrix of the
    unit difference vectors ``u = D / |D|``.  Working in the top principal
    subspace of the training directions, the generalised eigenproblem
    ``S_ant v = mu S_syn v`` yields directions ``v`` along which antonym
    differences carry ``mu`` times the energy of synonym differences.  If
    antonymy lived in a low-dimensional subspace, a handful of directions with
    ``mu`` far from 1 would classify the pairs.

    The classifier is the zero-mean Gaussian log-likelihood ratio restricted
    to the ``r`` directions with the largest ``|log mu|``::

        score(u) = 1/2 * sum_i [ z_i^2 (1 - 1/mu_i) - log mu_i ],   z = V^T u

    Returns the generalised eigenvalues (sorted by ``|log mu|``, largest
    first), the corresponding directions as columns of ``directions`` (in the
    coordinates of ``D``, unit length), the test accuracy as a function of the
    rank ``r`` and the test AUC of ``|u . v_1|`` for the leading direction.
    """
    from scipy.linalg import eigh

    y = np.asarray(y)
    U = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    U_train, y_train = U[train], y[train]

    # Principal subspace of the (uncentred, sign-symmetric) training directions.
    _, _, Vt = np.linalg.svd(U_train, full_matrices=False)
    basis = Vt[: min(n_components, Vt.shape[0])].T
    Z_train, Z_test = U_train @ basis, U[test] @ basis
    p = basis.shape[1]

    def moment(Z):
        S = Z.T @ Z / len(Z)
        return (1.0 - shrinkage) * S + shrinkage * np.trace(S) / p * np.eye(p)

    S_ant, S_syn = moment(Z_train[y_train == 1]), moment(Z_train[y_train == 0])
    mu, V = eigh(S_ant, S_syn)  # V^T S_syn V = I, V^T S_ant V = diag(mu)
    order = np.argsort(-np.abs(np.log(mu)))
    mu, V = mu[order], V[:, order]

    def scores(Z, r):
        proj = Z @ V[:, :r]
        return 0.5 * (proj ** 2 * (1.0 - 1.0 / mu[:r]) - np.log(mu[:r])).sum(axis=1)

    from sklearn.linear_model import LogisticRegression

    accuracy = {}
    for r in ranks:
        r = min(r, p)
        # Calibrate the decision threshold on the training scores.
        threshold = LogisticRegression().fit(scores(Z_train, r)[:, None], y_train)
        accuracy[int(r)] = float(threshold.score(scores(Z_test, r)[:, None], y[test]))
    from sklearn.metrics import roc_auc_score

    directions = basis @ V
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    leading = np.abs(U[test] @ directions[:, 0])
    auc = roc_auc_score(y[test], leading if mu[0] > 1 else -leading)
    return {"generalised_eigenvalues": mu, "directions": directions,
            "accuracy_by_rank": accuracy, "leading_direction_auc": float(auc)}


# ---------------------------------------------------------------------------
# Dataset structure
# ---------------------------------------------------------------------------

NEGATIVE_PREFIXES = ("un", "in", "im", "il", "ir", "dis", "non", "mis", "de", "anti", "a", "ab")


def is_derived_pair(a: str, b: str) -> bool:
    """True if one word is the other plus a negative prefix, or a -ful/-less pair.

    A deliberately simple string test for morphologically derived antonyms
    (*able/unable*, *honest/dishonest*, *careful/careless*).
    """
    a, b = a.lower(), b.lower()
    for longer, shorter in ((a, b), (b, a)):
        if any(longer in (prefix + shorter, prefix + "-" + shorter) for prefix in NEGATIVE_PREFIXES):
            return True
        if longer.endswith("less") and shorter.endswith("ful") and longer[:-4] == shorter[:-3]:
            return True
    return False


def pair_graph_summary(pairs: PairList) -> Dict[str, float]:
    """Word-reuse statistics of a list of pairs."""
    degree = Counter(w for pair in pairs for w in pair)
    counts = np.array(list(degree.values()))
    ordered = set(pairs)
    return {
        "n_pairs": len(pairs),
        "n_words": len(degree),
        "mean_pairs_per_word": float(counts.mean()),
        "max_pairs_per_word": int(counts.max()),
        "frac_words_in_one_pair": float((counts == 1).mean()),
        "frac_exact_duplicates": float(1.0 - len(ordered) / len(pairs)),
        "frac_with_reverse_present": float(np.mean([(b, a) in ordered for a, b in pairs])),
    }
