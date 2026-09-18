from __future__ import annotations
import math
from typing import Tuple, Dict, Any
import numpy as np



class NaiveBayesGaussian:

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.class_prior_log_: Dict[Any, float] = {}
        self.mean_: Dict[Any, np.ndarray] = {}
        self.var_: Dict[Any, np.ndarray] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> "NaiveBayesGaussian":

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        for c in self.classes_:
            Xc = X[y == c]

            self.class_prior_log_[c] = math.log(
                len(Xc) / len(X)
            )

            self.mean_[c] = Xc.mean(axis=0)

            self.var_[c] = (
                Xc.var(axis=0)
                + self.var_smoothing
            )

        return self

    def _log_gaussian_likelihood(
        self,
        x: np.ndarray,
        c: Any
    ) -> float:

        mu = self.mean_[c]
        var = self.var_[c]

        return (
            -0.5 * np.log(2 * np.pi * var).sum()
            -0.5 * (((x - mu) ** 2) / var).sum()
        )

    def _joint_log_likelihood(
        self,
        X: np.ndarray
    ) -> np.ndarray:

        scores = []

        for x in X:
            row = []

            for c in self.classes_:
                row.append(
                    self.class_prior_log_[c]
                    + self._log_gaussian_likelihood(x, c)
                )

            scores.append(row)

        return np.array(scores)

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:

        X = np.asarray(X, dtype=float)

        jll = self._joint_log_likelihood(X)

        idx = jll.argmax(axis=1)

        return self.classes_[idx]



class NaiveBayesMultinomial:

    def __init__(self, alpha: float = 1.0):

        if alpha < 0:
            raise ValueError("alpha must be >=0")

        self.alpha = alpha
        self.classes_: np.ndarray | None = None
        self.class_prior_log_: Dict[Any, float] = {}
        self.feature_log_prob_: Dict[Any, np.ndarray] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> "NaiveBayesMultinomial":

        X = np.asarray(X, dtype=float)

        if (X < 0).any():
            raise ValueError(
                "Multinomial NB requires non-negative counts."
            )

        y = np.asarray(y)

        self.classes_ = np.unique(y)

        for c in self.classes_:

            Xc = X[y == c]

            self.class_prior_log_[c] = math.log(
                len(Xc) / len(X)
            )

            counts = Xc.sum(axis=0)
            total = counts.sum()

            probs = (
                counts + self.alpha
            ) / (
                total + self.alpha * X.shape[1]
            )

            self.feature_log_prob_[c] = np.log(probs)

        return self

    def _joint_log_likelihood(
        self,
        X: np.ndarray
    ) -> np.ndarray:

        scores = []

        for x in X:

            row = []

            for c in self.classes_:

                row.append(
                    self.class_prior_log_[c]
                    + float(
                        np.dot(
                            x,
                            self.feature_log_prob_[c]
                        )
                    )
                )

            scores.append(row)

        return np.array(scores)

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:

        X = np.asarray(X, dtype=float)

        jll = self._joint_log_likelihood(X)

        idx = jll.argmax(axis=1)

        return self.classes_[idx]


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:

    return float(
        (np.asarray(y_true) == np.asarray(y_pred)).mean()
    )



def make_gaussian_blob_data(
    seed: int = 42
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray
]:

    rng = np.random.default_rng(seed)

    mean0 = np.array([0.0, 0.0])

    cov0 = np.array([
        [1.0, 0.4],
        [0.4, 1.2]
    ])

    mean1 = np.array([2.5, 2.0])

    cov1 = np.array([
        [1.1, -0.3],
        [-0.3, 1.0]
    ])

    X0 = rng.multivariate_normal(
        mean0,
        cov0,
        size=120
    )

    X1 = rng.multivariate_normal(
        mean1,
        cov1,
        size=120
    )

    y0 = np.zeros(
        len(X0),
        dtype=int
    )

    y1 = np.ones(
        len(X1),
        dtype=int
    )

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    idx = rng.permutation(len(X))

    split = int(0.75 * len(X))

    train_idx = idx[:split]
    test_idx = idx[split:]

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx]
    )




X_train, X_test, y_train, y_test = (
    make_gaussian_blob_data()
)

gaussian_model = NaiveBayesGaussian()

gaussian_model.fit(
    X_train,
    y_train
)

gaussian_pred = gaussian_model.predict(
    X_test
)

gaussian_accuracy = accuracy(
    y_test,
    gaussian_pred
)


print(
    f"[GaussianNB]Test accuracy:{gaussian_accuracy:.3f}"
)

print(
    "Forward_looking note:suitable for continuous "
    "features with class_conditional normality"
)

print()



vocabulary = [
    "buy",
    "discount",
    "meeting",
    "project"
]

X_text = np.array([
    [3, 2, 0, 0],
    [2, 3, 0, 0],
    [1, 2, 0, 0],
    [2, 1, 0, 0],

    [0, 0, 3, 2],
    [0, 0, 2, 3],
    [0, 0, 3, 1],
    [0, 0, 2, 2]
], dtype=float)

y_text = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1
])


X_text_train = X_text[:6]
X_text_test = X_text[6:]

y_text_train = y_text[:6]
y_text_test = y_text[6:]


multinomial_model = NaiveBayesMultinomial(
    alpha=1.0
)

multinomial_model.fit(
    X_text_train,
    y_text_train
)

multinomial_pred = multinomial_model.predict(
    X_text_test
)

multinomial_accuracy = accuracy(
    y_text_test,
    multinomial_pred
)


print(
    f"[MultinomialNB]Test accuracy:{multinomial_accuracy:.3f}"
)

print(
    f"Vocabulary:{vocabulary}"
)

print()

print(
    "thus,both Naive Bayes models were trained,"
    "evaluated,and executed successfully"
)
