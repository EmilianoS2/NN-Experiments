import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arch

# Load the frozen raw prices (regenerate with download_data.py if missing)
df = pd.read_csv("s&p500_daily.csv", index_col=0, parse_dates=True)

# --- base building block: log returns -------------------------------------
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df["sq_return"] = df["log_return"] ** 2   # squared return: kills sign, keeps magnitude

# ======================= THE TARGET (looks FORWARD) =======================
# RV_t for horizon h=1 is just |r_{t+1}| = sqrt(r_{t+1}^2).
# shift(-1) pulls TOMORROW's value back onto today's row.
df["target_rv"] = np.sqrt(df["sq_return"].shift(-1))
df["target_log_rv"] = np.log(df["target_rv"])   # predict log(RV), exponentiate later

# ======================= THE FEATURES (look BACKWARD) =====================
# Realized vol over the last N periods = sqrt( sum of last N squared returns ).
# rolling(N).sum() sweeps a window over the PAST and adds it up.
df["rv_5"] = np.sqrt(df["sq_return"].rolling(5).sum())   # <-- the pattern to copy

# ---- YOUR TURN: build the rest by copying the rv_5 pattern ----------------
df["rv_10"]  = np.sqrt(df["sq_return"].rolling(10).sum()) # realized vol, last 10
df["rv_21"]  = np.sqrt(df["sq_return"].rolling(21).sum()) # realized vol, last 21
df["abs_ret"] = np.sqrt(df["sq_return"]) # most recent |r_t|  (no rolling needed)
df["mean_5"] = df["log_return"].rolling(5).mean() # mean log_return over last 5  (.rolling(5).mean())
df["mean_10"] = df["log_return"].rolling(10).mean()
df["skew_21"] = df["log_return"].rolling(21).skew() # .rolling(21).skew()
df["kurt_21"] = df["log_return"].rolling(21).kurt() # .rolling(21).kurt()
df["vol_diff"] = df["rv_5"] - df["rv_21"] # rv_5 - rv_21  (is vol rising or falling?)

print(df.tail())
plt.plot(df["rv_21"])
print(arch.__version__)
