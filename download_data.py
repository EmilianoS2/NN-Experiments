"""Run this ONCE to fetch raw prices and freeze them to CSV.
Day-to-day work loads the CSV; this script only regenerates it."""

import yfinance as yf

df = yf.download("^GSPC", period="max", interval="1d")

# yfinance returns two stacked column levels: (Price, Ticker) e.g. ("Close", "^GSPC").
# Collapse to a single level so the saved CSV has one clean header row.
df.columns = df.columns.get_level_values(0)
df.columns.name = None

df = df.loc["1962":]          # keep 1950 onward (cleaner, modern market structure)
df.to_csv("s&p500_daily.csv")

print(df.head())
print("rows:", len(df))
