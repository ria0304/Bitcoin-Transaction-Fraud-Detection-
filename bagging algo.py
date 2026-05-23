"""
Bagging (Bootstrap Aggregating) - Step-by-step Demo
Uses the EXACT bootstrap samples from the lecture slide.
"""

import numpy as np
from collections import Counter

# ──────────────────────────────────────────────
# 0.  Original dataset  (from the slide)
#     x : 0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
#     y :  +1   +1   +1   -1   -1   -1   -1   -1   +1   +1
# ──────────────────────────────────────────────
X_orig = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y_orig = np.array([  1,   1,   1,  -1,  -1,  -1,  -1,  -1,   1,   1])

# Label lookup from original dataset
label_map = dict(zip(X_orig, y_orig))

# ──────────────────────────────────────────────
# 1.  Exact bootstrap samples from the slide
# ──────────────────────────────────────────────
BOOTSTRAP_SAMPLES = [
    # Round 1  → expected rule: x <= 0.35 → y=+1, x > 0.35 → y=-1
    [0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.9, 0.9],
    # Round 2  → expected rule: x <= 0.7  → y=+1, x > 0.7  → y=+1
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.9, 1.0, 1.0, 1.0],
    # Round 3  → expected rule: x <= 0.35 → y=+1, x > 0.35 → y=-1
    [0.1, 0.2, 0.3, 0.4, 0.4, 0.5, 0.7, 0.7, 0.8, 0.9],
    # Round 4  → expected rule: x <= 0.3  → y=+1, x > 0.3  → y=-1
    [0.1, 0.1, 0.2, 0.4, 0.4, 0.5, 0.5, 0.7, 0.8, 0.9],
    # Round 5  → expected rule: x <= 0.35 → y=+1, x > 0.35 → y=-1
    [0.1, 0.1, 0.2, 0.5, 0.6, 0.6, 0.6, 1.0, 1.0, 1.0],
]


# ──────────────────────────────────────────────
# 2.  Decision Stump  (1-level decision tree)
# ──────────────────────────────────────────────
class DecisionStump:
    def __init__(self):
        self.threshold   = None
        self.left_label  = None   # label when x <= threshold
        self.right_label = None   # label when x >  threshold

    def fit(self, X, y):
        best_err = float('inf')
        candidates = np.unique(X)

        for i in range(len(candidates) - 1):
            thresh = round((candidates[i] + candidates[i + 1]) / 2, 4)

            for left_lbl in [1, -1]:
                right_lbl = -left_lbl
                preds = np.where(X <= thresh, left_lbl, right_lbl)
                err   = np.mean(preds != y)
                if err < best_err:
                    best_err         = err
                    self.threshold   = thresh
                    self.left_label  = left_lbl
                    self.right_label = right_lbl

    def predict(self, X):
        return np.where(X <= self.threshold,
                        self.left_label,
                        self.right_label)

    def rule_str(self):
        return (f"x <= {self.threshold} → y = {self.left_label:+d}  |  "
                f"x >  {self.threshold} → y = {self.right_label:+d}")


# ──────────────────────────────────────────────
# 3.  Bagging  (Bootstrap Aggregating)
# ──────────────────────────────────────────────
class Bagging:
    def __init__(self, bootstrap_samples=None):
        self.bootstrap_samples = bootstrap_samples
        self.stumps_ = []

    def fit(self, label_lookup):
        print("=" * 68)
        print("  BAGGING — TRAINING PHASE")
        print("=" * 68)

        for r, xs in enumerate(self.bootstrap_samples):
            X_b = np.array(xs)
            y_b = np.array([label_lookup[x] for x in xs])

            stump = DecisionStump()
            stump.fit(X_b, y_b)
            self.stumps_.append(stump)

            x_str = "  ".join(f"{v:.1f}" for v in X_b)
            y_str = "  ".join(f"{int(v):+d}" for v in y_b)
            print(f"\n  Round {r + 1}:")
            print(f"    x : {x_str}")
            print(f"    y : {y_str}")
            print(f"    Rule → {stump.rule_str()}")

        print()

    def predict(self, X, verbose=True):
        votes = np.array([s.predict(X) for s in self.stumps_])

        if verbose:
            print("=" * 68)
            print("  BAGGING — PREDICTION PHASE  (majority vote)")
            print("=" * 68)
            print(f"\n  {'x':>5}  {'Votes (R1–R5)':^27}  {'Final':>6}")
            print("  " + "-" * 48)
            for i, xi in enumerate(X):
                vote_row = votes[:, i]
                tally    = Counter(vote_row)
                final    = max(tally, key=tally.get)
                v_str    = "  ".join(f"{int(v):+d}" for v in vote_row)
                print(f"  {xi:>5.2f}  [{v_str}]   → {final:+d}")
            print()

        return np.array([
            max(Counter(votes[:, i]), key=Counter(votes[:, i]).get)
            for i in range(votes.shape[1])
        ])


# ──────────────────────────────────────────────
# 4.  Run
# ──────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "=" * 68)
    print("  ORIGINAL DATASET")
    print("=" * 68)
    print(f"  x : {'  '.join(f'{v:.1f}' for v in X_orig)}")
    print(f"  y : {'  '.join(f'{int(v):+d}' for v in y_orig)}\n")

    model = Bagging(bootstrap_samples=BOOTSTRAP_SAMPLES)
    model.fit(label_map)

    y_pred = model.predict(X_orig, verbose=True)

    acc = np.mean(y_pred == y_orig) * 100
    print("=" * 68)
    print(f"  Final predictions : {[int(v) for v in y_pred]}")
    print(f"  True labels       : {list(y_orig)}")
    print(f"  Accuracy          : {acc:.1f}%")
    print("=" * 68)

    print("\n  Predicting a new point  x = 0.45 ...")
    y_new = model.predict(np.array([0.45]), verbose=True)
    print(f"  Predicted label for x = 0.45 : {int(y_new[0]):+d}\n")
