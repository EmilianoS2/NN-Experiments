# extrapolation/steps_data.py
# Truth: you take one step every 3 seconds.  steps = seconds / 3
import numpy as np

RATE = 3.0
TRAIN_MAX = 15_000.0          # 15,000 seconds == 5,000 steps

rng = np.random.default_rng(0)

def truth(x):
    return x / RATE

# Splits ===================================================================
X_train  = rng.uniform(0, TRAIN_MAX, 5000)                     # what it sees
X_interp = rng.uniform(0, TRAIN_MAX, 1000)                     # CONTROL: inside the range
X_near   = rng.uniform(TRAIN_MAX, TRAIN_MAX * 1.5, 2500)       # 15k -> 22.5k seconds
X_far    = np.array([30_000.0, 50_000.0, 100_000.0])           # far outside

y_train, y_interp, y_near, y_far = (truth(v) for v in (X_train, X_interp, X_near, X_far))

# Normalization (TRAINING statistics only -- never refit on test) ==========
x_mu, x_sd = X_train.mean(), X_train.std()
y_mu, y_sd = y_train.mean(), y_train.std()

def nx(x): return (x - x_mu) / x_sd      # raw seconds  -> model input
def ny(y): return (y - y_mu) / y_sd      # raw steps    -> model target
def dy(y): return y * y_sd + y_mu        # model output -> raw steps

# Scoring ==================================================================
def evaluate(name, predict):
    """predict: raw seconds (np array) -> raw predicted steps (np array)"""
    print(f"\n=== {name} ===")
    for label, x, y in [("interpolation  (0-15k)", X_interp, y_interp),
                        ("near extrap (15k-22.5k)", X_near, y_near)]:
        err = np.abs(predict(x) - y)
        print(f"  {label:24s}  MAE {err.mean():12,.2f}   worst {err.max():12,.2f}")
    print("  far extrapolation:")
    for x, y in zip(X_far, y_far):
        p = float(predict(np.array([x]))[0])
        print(f"    {x:9,.0f}s   true {y:10,.1f}   pred {p:10,.1f}   off by {abs(p - y):9,.1f}")
