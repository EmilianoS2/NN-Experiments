"""Shared feature engineering — one source of truth for every model script.

Usage:
    import pandas as pd
    from features import build_features
    df = build_features(pd.read_csv("s&p500_daily.csv", index_col=0, parse_dates=True))
"""

import numpy as np


def build_features(df):
    """Add log returns, the target, and the 8 backward-looking features to df."""
    # base building blocks
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["sq_return"] = df["log_return"] ** 2
    
    # TARGET (looks forward): h=5 realized vol, then its log
    df["target_rv"] = np.sqrt(df["sq_return"].rolling(5).sum().shift(-5))
    df["target_log_rv"] = np.log(df["target_rv"])

    # FEATURES (look backward)
    df["rv_5"] = np.sqrt(df["sq_return"].rolling(5).sum())
    df["rv_10"] = np.sqrt(df["sq_return"].rolling(10).sum())
    df["rv_21"] = np.sqrt(df["sq_return"].rolling(21).sum())
    df["abs_ret"] = np.sqrt(df["sq_return"])
    df["mean_5"] = df["log_return"].rolling(5).mean()
    df["mean_10"] = df["log_return"].rolling(10).mean()
    df["skew_21"] = df["log_return"].rolling(21).skew()
    df["kurt_21"] = df["log_return"].rolling(21).kurt()
    df["vol_diff"] = df["rv_5"] - df["rv_21"]

    return df
