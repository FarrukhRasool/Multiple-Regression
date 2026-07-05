"""
framework.py — Task 2 rule engine for `target02` (problem 41).

Instead of a black-box model, `target02` is predicted with a small, human-readable
set of `(condition, calculation)` rules extracted in `task_2.ipynb`. The engine walks
each row and applies the calculation of the *first* rule whose condition is satisfied.

Extracted rules
---------------
    feat_187 <= 0.7 :  target = -0.173*feat_187 - 1.594*feat_64 + 0.358*feat_126 - 0.571*feat_53 + 0.089
    feat_187 >  0.7 :  target =  0.15*feat_64 + 1.85*feat_126 + 1.05*feat_53

Usage
-----
    python framework.py --eval_file_path problem_41/EVAL_41.csv
"""

import argparse
import operator

import pandas as pd

# Feature-name -> column index in the dataset (see task_2.ipynb).
FEAT_187, FEAT_64, FEAT_126, FEAT_53 = 187, 64, 126, 53

OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def framework(pairs, arr):
    """Apply the first matching rule to every row of ``arr``.

    Args:
        pairs: list of ``(condition, calc)`` tuples. ``calc`` is a callable that
            maps a feature row to a scalar prediction.
        arr: numpy array whose columns are the features in order ``feat_0, feat_1, ...``.

    Returns:
        A list with one prediction per row. Rows matching no condition are skipped.
    """
    targets = []
    for i in range(arr.shape[0]):
        row = arr[i]
        for cond, calc in pairs:
            if cond_eval(cond, row):
                targets.append(calc(row))
                break
    return targets


def cond_eval(condition, arr):
    """Evaluate a single condition against a feature row.

    Args:
        condition: a ``(int, str, float)`` tuple of ``(feature_index, operator, threshold)``,
            where ``operator`` is one of the keys in ``OPS``. ``None`` is always True.
        arr: the feature row the condition is tested on.
    """
    if condition is None:
        return True
    feature_index, op, threshold = condition
    return OPS[op](arr[feature_index], threshold)


def build_rules():
    """Return the extracted `(condition, calculation)` rule set for `target02`."""

    def calc_low(arr):
        """feat_187 <= 0.7"""
        return (-0.173 * arr[FEAT_187] - 1.594 * arr[FEAT_64]
                + 0.358 * arr[FEAT_126] - 0.571 * arr[FEAT_53] + 0.089)

    def calc_high(arr):
        """feat_187 > 0.7"""
        return 0.15 * arr[FEAT_64] + 1.85 * arr[FEAT_126] + 1.05 * arr[FEAT_53]

    return [
        ((FEAT_187, "<=", 0.7), calc_low),
        ((FEAT_187, ">", 0.7), calc_high),
    ]


def main(args):
    data_array = pd.read_csv(args.eval_file_path).values
    return framework(build_rules(), data_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2 rule engine for target02")
    parser.add_argument("--eval_file_path", required=True, help="Path to EVAL_<ID>.csv")
    args = parser.parse_args()

    target02 = main(args)
    print(target02)