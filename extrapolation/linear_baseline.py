# extrapolation/linear_baseline.py
import numpy as np
from steps_data import X_train, y_train, evaluate

# The model's SHAPE is the truth's shape, so there is nothing left to get wrong.
A = np.vstack([X_train, np.ones_like(X_train)]).T
w, b = np.linalg.lstsq(A, y_train, rcond=None)[0]

print(f"fitted:  steps = {w:.6f} * seconds + {b:.6f}   (truth: {1/3:.6f}, 0)")
evaluate("least squares (y = wx + b)", lambda x: w * x + b)
