import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from features import build_features


# Data Prep
# --------------------------------------------------------------------------------------------------------------------
df = build_features(pd.read_csv("s&p500_daily.csv", index_col=0, parse_dates=True))

cols = ["rv_5","rv_10","rv_21","abs_ret","mean_5","mean_10","skew_21","kurt_21","target_log_rv"]
clean = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

X = clean[["rv_5", "rv_10", "rv_21", "abs_ret", "mean_5", "mean_10", "skew_21", "kurt_21"]]
Y = clean["target_log_rv"]

n = len(X)
x_train = X.iloc[:int(n * 0.7)]
x_validate = X.iloc[int(n * 0.7):int(n * 0.85)]
x_test = X.iloc[int(n * 0.85):]

k = len(Y)
y_train = Y.iloc[:int(k * 0.7)]
y_validate = Y.iloc[int(k * 0.7):int(k * 0.85)]
y_test = Y.iloc[int(k * 0.85):]

mean = x_train.mean()
std = x_train.std()

x_train = (x_train - mean) / std
x_validate = (x_validate - mean) / std
x_test = (x_test - mean) / std

# Tensors =============================================================================================================

x_train_t = torch.tensor(x_train.values, dtype=torch.float32)
x_validate_t = torch.tensor(x_validate.values, dtype=torch.float32)
x_test_t = torch.tensor(x_test.values, dtype=torch.float32)

y_train_t = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_validate_t = torch.tensor(y_validate.values, dtype=torch.float32).reshape(-1, 1)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

print(x_train_t.shape)

# Network ===============================================================================================================

model = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Training ==============================================================================================================

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

epochs = 500

for epoch in range(epochs):
    # TRAIN
    model.train()
    optimizer.zero_grad()
    pred = model(x_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    optimizer.step()

    # EVALUATE ON VALIDATION
    if epoch % 50 == 0:
        model.eval()                               # eval mode for a clean val read
        with torch.no_grad():
            val_loss = loss_fn(model(x_validate_t), y_validate_t)
        print(f"epoch {epoch:4d}   train {loss.item():.4f}   val {val_loss.item():.4f}")

# EVALS =================================================================================================================

model.eval()
with torch.no_grad():
    pred_log = model(x_test_t)

pred_vol = torch.exp(pred_log) * 100
truth_vol = torch.exp(y_test_t) * 100

rmse = torch.sqrt(((pred_vol - truth_vol) ** 2).mean())
print("MLP test RMSE:", rmse.item())