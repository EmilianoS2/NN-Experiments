import numpy as np
import torch
import torch.nn as nn
from primes_gaps import X_train, X_val, X_test, y_train, y_val, y_test, build_features, score

# Features =================================================================
F_train = build_features(X_train)
F_val   = build_features(X_val)
F_test  = build_features(X_test)

# Normalize (training statistics only) =====================================
mean, std = F_train.mean(axis=0), F_train.std(axis=0)
F_train = (F_train - mean) / std
F_val   = (F_val - mean) / std
F_test  = (F_test - mean) / std

# Tensors ==================================================================
X_train_t = torch.tensor(F_train, dtype=torch.float32)
X_val_t = torch.tensor(F_val, dtype=torch.float32)
X_test_t = torch.tensor(F_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Network =====================================================================

model = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Training ========================================================================

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

epochs = 300

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
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t)
        print(f"epoch {epoch:4d}   train {loss.item():.4f}   val {val_loss.item():.4f}")

# EVALS ==========================================================================

model.eval()
with torch.no_grad():
    pred = model(X_test_t)

rmse = torch.sqrt(((pred - y_test_t) ** 2).mean())

score("MLP", pred.numpy(), y_test)
