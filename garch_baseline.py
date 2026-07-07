from arch import arch_model
import pandas as pd
import numpy as np


df = pd.read_csv("s&p500_daily.csv", index_col=0, parse_dates=True)
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
returns = df["log_return"].dropna()
returns = returns * 100

n = len(returns)
train = returns.iloc[:int(n*0.7)]
validate = returns.iloc[int(n*0.7):int(n*0.85)]
test = returns.iloc[int(n*0.85):]

am  = arch_model(returns, vol="Garch", p=1, q=1)   # p=1, q=1 -> the "(1,1)"
res = am.fit(last_obs=test.index[0], disp="off")
fc = res.forecast(start=test.index[0], horizon=5, reindex=False)
vol_forecast = np.sqrt(fc.variance.sum(axis=1))   # <- the one-step-ahead column name

# --- SCORE: how far are the forecasts from what actually happened? ---------
sq = returns ** 2
realized = np.sqrt(sq.rolling(5).sum().shift(-5))          # |r_{t+1}| = tomorrow's move size (h=5 realized vol)
realized = realized.loc[vol_forecast.index]   # keep only the test days, aligned to the forecasts

rmse = np.sqrt(((vol_forecast - realized) ** 2).mean())   # avg distance: forecast vs truth
print("GARCH test RMSE:", rmse)

print(res.summary())
print(len(train)+len(validate)+len(test) == n)
print(vol_forecast.head())
print(len(vol_forecast), len(test))   # these two numbers should match
print(len(vol_forecast) == len(test))

# 0.81