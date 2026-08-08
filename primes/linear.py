# primes/linear.py
import numpy as np
from primes_gaps import X_train, X_test, y_train, y_test, build_features, score

A = np.hstack([build_features(X_train), np.ones((len(X_train), 1))])
B = np.hstack([build_features(X_test),  np.ones((len(X_test), 1))])

w = np.linalg.lstsq(A, y_train, rcond=None)[0]
score("linear (gaps + ln p)", B @ w, y_test)
