"""
surrogate_optim.py
==================
Surrogate-model-guided optimisation over a discrete hyperparameter space.

Algorithm
---------
1. Evaluate a random *initial batch* of `initial_batch` parameter combinations.
2. Fit a RandomForest surrogate on (params → Sharpe).
3. Score all unevaluated candidates with the surrogate.
4. Evaluate the top `subsequent_batch` predicted candidates (actual strategy run).
5. Refit the surrogate on the enlarged dataset.
6. Repeat until the best Sharpe does not improve for `patience` consecutive rounds,
   or the candidate pool is exhausted.

The surrogate dramatically reduces the number of full strategy evaluations needed
compared to an exhaustive grid search while still concentrating compute on the
most promising regions of the parameter space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, List, Tuple, Any, Optional
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor


# ── helpers ───────────────────────────────────────────────────────────────────

def _encode_args(args: List[Tuple]) -> np.ndarray:
    """
    Convert a list of parameter tuples to a numeric feature matrix.

    Encoding rules:
      False  → -1.0  (disabled / not set)
      True   →  1.0
      None   →  0.0
      number → float(value)
    """
    rows = []
    for a in args:
        row = []
        for v in a:
            if v is False:
                row.append(-1.0)
            elif v is True:
                row.append(1.0)
            elif v is None:
                row.append(0.0)
            else:
                row.append(float(v))
        rows.append(row)
    return np.array(rows, dtype=float)


def _sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe ratio from a daily return series."""
    s = returns.std()
    if s == 0 or np.isnan(s):
        return 0.0
    return float(returns.mean() * 365 / (s * np.sqrt(365)))


# ── main function ─────────────────────────────────────────────────────────────

def surrogate_optimise(
    all_args: List[Tuple],
    run_fn: Callable[..., pd.Series],
    initial_batch: int = 2000,
    subsequent_batch: int = 500,
    patience: int = 2,
    n_jobs: int = 6,
    n_estimators: int = 200,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[List[Tuple], List[pd.Series], List[float], int]:
    """
    Surrogate-model-guided optimisation over a discrete parameter space.

    Parameters
    ----------
    all_args          : Full list of parameter tuples to search over.
    run_fn            : Callable(*arg) → pd.Series of daily returns.
    initial_batch     : Number of randomly-drawn candidates in round 0.
    subsequent_batch  : Number of surrogate-selected candidates in each
                        subsequent round.
    patience          : Stop after this many consecutive rounds with no
                        improvement in best Sharpe.
    n_jobs            : Parallel workers (passed to joblib).
    n_estimators      : Trees in the RandomForest surrogate.
    random_state      : RNG seed for reproducibility.
    verbose           : Print round-by-round progress.

    Returns
    -------
    evaluated_args    : Parameter tuples that were actually run.
    evaluated_returns : Corresponding pd.Series of daily returns.
    evaluated_sharpes : Corresponding annualised Sharpe ratios.
    best_idx          : Index into the above lists for the best strategy found.
    """
    rng = np.random.default_rng(random_state)
    all_args = list(all_args)
    n_total = len(all_args)

    evaluated_args: List[Tuple] = []
    evaluated_returns: List[pd.Series] = []
    evaluated_sharpes: List[float] = []

    # Track which original indices are still unevaluated
    remaining_indices = list(range(n_total))

    def _eval_batch(indices: List[int]) -> List[pd.Series]:
        batch_args = [all_args[i] for i in indices]
        results: List[pd.Series] = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_fn)(*a) for a in batch_args
        )
        return results

    # ── Round 0: random initial batch ─────────────────────────────────────────
    init_size = min(initial_batch, len(remaining_indices))
    chosen_local = rng.choice(len(remaining_indices), size=init_size, replace=False).tolist()
    chosen = [remaining_indices[j] for j in chosen_local]

    if verbose:
        print(f"[surrogate] Round 0 — evaluating {len(chosen):,} random candidates …")

    rets = _eval_batch(chosen)
    sharpes = [_sharpe(r) for r in rets]

    evaluated_args.extend([all_args[i] for i in chosen])
    evaluated_returns.extend(rets)
    evaluated_sharpes.extend(sharpes)

    chosen_set = set(chosen)
    remaining_indices = [i for i in remaining_indices if i not in chosen_set]

    best_so_far = max(evaluated_sharpes)
    no_improve_count = 0
    round_num = 1

    if verbose:
        print(f"[surrogate] Round 0 done  |  best Sharpe: {best_so_far:.4f}"
              f"  |  evaluated: {len(evaluated_args):,}/{n_total:,}")

    # ── Subsequent rounds: surrogate-guided ───────────────────────────────────
    while remaining_indices and no_improve_count < patience:

        # Fit surrogate on all data collected so far
        X_train = _encode_args(evaluated_args)
        y_train = np.array(evaluated_sharpes, dtype=float)

        surrogate = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=3,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        surrogate.fit(X_train, y_train)

        # Predict Sharpe for every remaining candidate
        remaining_args = [all_args[i] for i in remaining_indices]
        X_remaining = _encode_args(remaining_args)
        predicted = surrogate.predict(X_remaining)

        # Select top `subsequent_batch` by predicted Sharpe
        n_pick = min(subsequent_batch, len(remaining_indices))
        top_local = np.argsort(predicted)[::-1][:n_pick]
        chosen = [remaining_indices[j] for j in top_local]

        if verbose:
            print(
                f"[surrogate] Round {round_num} — evaluating {len(chosen):,} "
                f"surrogate-selected candidates "
                f"(top predicted Sharpe: {predicted[top_local[0]]:.4f}) …"
            )

        rets = _eval_batch(chosen)
        sharpes = [_sharpe(r) for r in rets]

        evaluated_args.extend([all_args[i] for i in chosen])
        evaluated_returns.extend(rets)
        evaluated_sharpes.extend(sharpes)

        chosen_set = set(chosen)
        remaining_indices = [i for i in remaining_indices if i not in chosen_set]

        new_best = max(evaluated_sharpes)
        if new_best > best_so_far:
            best_so_far = new_best
            no_improve_count = 0
            marker = "✓ improved"
        else:
            no_improve_count += 1
            marker = f"no improvement ({no_improve_count}/{patience})"

        if verbose:
            print(
                f"[surrogate] Round {round_num} done  |  best Sharpe: {best_so_far:.4f}"
                f"  |  {marker}"
                f"  |  evaluated: {len(evaluated_args):,}/{n_total:,}"
            )

        round_num += 1

    best_idx = int(np.argmax(evaluated_sharpes))

    if verbose:
        print(
            f"\n[surrogate] Finished after {round_num} round(s).\n"
            f"  Evaluated : {len(evaluated_args):,} / {n_total:,} candidates\n"
            f"  Best Sharpe: {evaluated_sharpes[best_idx]:.4f}  (index {best_idx})"
        )

    return evaluated_args, evaluated_returns, evaluated_sharpes, best_idx
