"""
Honest baselines for win-probability evaluation.

Each baseline is trained and evaluated on the *same* information set as the
model it competes with.  Using a baseline trained on pregame data to evaluate
against a live-prefix model is not an honest comparison and is explicitly
avoided here.

Baselines implemented
---------------------
PregameLogisticBaseline
    Logistic regression on pregame features only (team win records, home
    indicator).  Trained on the train split of pregame_context.parquet.
    Predicts P(home wins) from pre-game information.

LivePrefixScoreBaseline
    Logistic regression on (score_diff, inning, is_top) at each half-inning
    boundary.  Trained on the train split of prefix_states.parquet.
    Predicts P(home wins) given observed game state at a boundary.

Both baselines expose a fit(train_df) / predict(df) → np.ndarray interface.
They are designed to be run by eval/evaluate.py, not imported directly.

Matching information sets
-------------------------
For each evaluation mode the correct baseline is:

    pregame  + MC/MCMC  →  PregameLogisticBaseline
    live-prefix + MC/MCMC → LivePrefixScoreBaseline

Evaluating a live-prefix model against PregameLogisticBaseline would be
unfair (baseline uses less information).  Evaluating against
LivePrefixScoreBaseline with the same conditioning is the honest comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class _LogisticRegression:
    """Minimal logistic regression via gradient descent (no sklearn dependency).

    Args:
        lr:      Learning rate.
        n_iter:  Number of gradient descent iterations.
        l2:      L2 regularisation coefficient.
    """

    def __init__(self, lr: float = 0.1, n_iter: int = 1000, l2: float = 1e-4):
        self.lr     = lr
        self.n_iter = n_iter
        self.l2     = l2
        self.w: np.ndarray = np.array([])
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogisticRegression":
        n, d    = X.shape
        self.w  = np.zeros(d)
        self.b  = 0.0
        for _ in range(self.n_iter):
            logits = X @ self.w + self.b
            probs  = _sigmoid(logits)
            err    = probs - y
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.w + self.b)


# ---------------------------------------------------------------------------
# Pregame baseline
# ---------------------------------------------------------------------------

class PregameLogisticBaseline:
    """Logistic regression on pregame information (matching information set).

    Features used (all available pre-first-pitch):
        - home_team_win_pct: Home team win percentage entering the game (causal)
        - away_team_win_pct: Away team win percentage entering the game (causal)
        - win_pct_diff:      home_team_win_pct - away_team_win_pct (computed internally)

    Column names match data/features.py build_team_records() output.
    If a column is missing, it is filled with 0.5 (neutral prior).
    """

    _FEATURE_COLS = ["home_team_win_pct", "away_team_win_pct"]

    def __init__(self) -> None:
        self._model = _LogisticRegression(lr=0.5, n_iter=2000, l2=1e-4)
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "PregameLogisticBaseline":
        """Fit on training-split pregame_context rows.

        Args:
            train_df: DataFrame with pregame features and 'home_win' column.
        """
        X = self._build_features(train_df)
        y = train_df["home_win"].astype(float).to_numpy()
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict P(home wins) for each row.

        Args:
            df: DataFrame with pregame feature columns.

        Returns:
            1-D array of win probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        X = self._build_features(df)
        return self._model.predict_proba(X)

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self._FEATURE_COLS:
            if col in df.columns:
                cols.append(df[col].fillna(0.5).to_numpy())
            else:
                cols.append(np.full(len(df), 0.5))
        # Append win_pct_diff computed from the two win-pct columns.
        cols.append(cols[0] - cols[1])
        return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Live-prefix baseline (same information set as live-prefix model)
# ---------------------------------------------------------------------------

class LivePrefixScoreBaseline:
    """Logistic regression on live-game prefix state (matching information set).

    Features used (at each half-inning boundary):
        - score_diff:   home_score - away_score at the boundary
        - inning:       current inning (1-indexed)
        - is_top:       1 if top of inning (away batting), 0 otherwise
        - half_innings_played: number of half-innings completed

    These columns are expected in prefix_states.parquet as produced by
    data/tables.py (Option A: half-inning boundary conditioning).
    """

    _FEATURE_COLS = ["score_diff", "inning", "is_top", "half_innings_played"]

    def __init__(self) -> None:
        self._model = _LogisticRegression(lr=0.3, n_iter=2000, l2=1e-4)
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "LivePrefixScoreBaseline":
        """Fit on training-split prefix_states rows.

        Args:
            train_df: DataFrame with prefix features and 'home_win' column.
        """
        X = self._build_features(train_df)
        y = train_df["home_win"].astype(float).to_numpy()
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict P(home wins) for each row.

        Args:
            df: DataFrame with prefix feature columns.

        Returns:
            1-D array of win probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        X = self._build_features(df)
        return self._model.predict_proba(X)

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        parts = []

        # score_diff
        if "score_diff" in df.columns:
            parts.append(df["score_diff"].fillna(0.0).to_numpy())
        elif "home_score" in df.columns and "away_score" in df.columns:
            parts.append((df["home_score"] - df["away_score"]).fillna(0.0).to_numpy())
        else:
            parts.append(np.zeros(len(df)))

        # inning (normalised to [0,1] over 9 innings)
        if "inning" in df.columns:
            parts.append((df["inning"].fillna(1.0).to_numpy() - 1) / 9.0)
        else:
            parts.append(np.zeros(len(df)))

        # is_top
        if "is_top" in df.columns:
            parts.append(df["is_top"].astype(float).fillna(0.0).to_numpy())
        else:
            parts.append(np.zeros(len(df)))

        # half_innings_played
        if "half_innings_played" in df.columns:
            parts.append(df["half_innings_played"].fillna(0.0).to_numpy() / 18.0)
        elif "prefix_half_innings" in df.columns:
            parts.append(df["prefix_half_innings"].fillna(0.0).to_numpy() / 18.0)
        else:
            parts.append(np.zeros(len(df)))

        return np.column_stack(parts)
