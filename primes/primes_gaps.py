import sympy
import numpy as np
primes = list(sympy.primerange(2, 110000))

k = 5

X, y = [], []
for i in range(k, len(primes)):
    X.append(primes[i-k:i])
    y.append(primes[i] - primes[i - 1])

X = np.array(X)
y = np.array(y)

np.random.seed(42)
idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]

n = len(X)
X_train = X[:int(n*0.7)]
X_val = X[int(n*0.7):int(n*0.85)]
X_test = X[int(n*0.85):]

y_train = y[:int(n*0.7)]
y_val = y[int(n*0.7):int(n*0.85)]
y_test = y[int(n*0.85):]

def build_features(X):
    gaps = np.diff(X, axis=1)              # (n, 4) — local structure
    logp = np.log(X[:, -1:])               # (n, 1) — magnitude / PNT trend
    return np.hstack([gaps, logp])         # (n, 5)

def score(name, pred, truth):
    pred = np.asarray(pred, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()
    mae   = np.abs(pred - truth).mean()
    rmse  = np.sqrt(((pred - truth) ** 2).mean())
    exact = (np.round(pred / 2) * 2 == truth).mean()
    print(f"{name:24s}  MAE {mae:.3f}   RMSE {rmse:.3f}   exact {exact:.1%}")
