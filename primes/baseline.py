import numpy as np
from primes_gaps import X_test, y_test, score

pred_a = np.diff(X_test, axis=1).mean(axis=1)
pred_b = np.log(X_test[:, -1])

score("avg recent gap", pred_a, y_test)
score("ln(p) — PNT", pred_b, y_test)
score("always 6", np.full(len(y_test), 6), y_test)
