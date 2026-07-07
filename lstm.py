import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from features import build_features

# DATA PREP =========================================================================================================

df = build_features(pd.read_csv("s&p500_daily.csv", index_col=0, parse_dates=True))
cols = ['log_return', 'target_log_rv']
clean = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

N = 30
returns = clean["log_return"].values      # flat array of all returns
targets = clean["target_log_rv"].values

X_seq, y_seq = [], []
for i in range(N, len(clean)):
    X_seq.append(returns[i-N:i])
    y_seq.append(targets[i-1])   

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

k = len(X_seq)
X_seq_train = X_seq[:int(k * 0.7)]
X_seq_validate = X_seq[int(k * 0.7):int(k * 0.85)]
X_seq_test = X_seq[int(k * 0.85):]

y_seq_train = y_seq[:int(k * 0.7)]
y_seq_validate = y_seq[int(k * 0.7):int(k * 0.85)]
y_seq_test = y_seq[int(k * 0.85):]

mean = X_seq_train.mean()
std = X_seq_train.std()

X_train = (X_seq_train - mean) / std
X_val = (X_seq_validate - mean) /std
X_test = (X_seq_test - mean) /std

# TENSORS =========================================================================================================

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

y_train_t = torch.tensor(y_seq_train, dtype=torch.float32).reshape(-1, 1)
y_val_t = torch.tensor(y_seq_validate, dtype=torch.float32).reshape(-1, 1)
y_test_t = torch.tensor(y_seq_test, dtype=torch.float32).reshape(-1, 1)

# NETWORK ==========================================================================================================

class VolLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1     = nn.Linear(32, 16)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(16, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)     # x:(N,30,1) → out:(N,30,32)
        last = out[:, -1, :]           # last timestep only → (N,32)
        z = self.dropout(last)         # randomly zero 20% (train only)
        z = self.relu(self.fc1(z))     # 32 → 16, ReLU
        z = self.fc2(z)                # 16 → 1, linear output
        return z

model = VolLSTM()

# TRAINING ==========================================================================================================

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

epochs = 600

for epoch in range(epochs):
    # TRAIN
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    optimizer.step()

    # EVALUATE ON VALIDATION
    if epoch % 50 == 0:
        model.eval()                               # eval mode for a clean val read
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t)
        print(f"epoch {epoch:4d}   train {loss.item():.4f}   val {val_loss.item():.4f}")

# EVALS =============================================================================================================

model.eval()
with torch.no_grad():
    pred_log = model(X_test_t)
pred_vol  = torch.exp(pred_log) * 100
truth_vol = torch.exp(y_test_t) * 100
rmse = torch.sqrt(((pred_vol - truth_vol) ** 2).mean())
print("LSTM test RMSE:", rmse.item())