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
X_train_t = torch.tensor(F_train, dtype=torch.float32).unsqueeze(-1)
X_val_t = torch.tensor(F_val, dtype=torch.float32).unsqueeze(-1)
X_test_t = torch.tensor(F_test, dtype=torch.float32).unsqueeze(-1)

y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Network ======================================================================

class PrimeTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, layers=1):
        super().__init__()
        self.project = nn.Linear(1, d_model)                    # scalar -> token vector
        self.pos = nn.Parameter(torch.zeros(1, 5, d_model))     # learned position info
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64,
                                         batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                   # x: (n, 5, 1)
        h = self.project(x) + self.pos      # (n, 5, 32)  — tokens + position
        h = self.encoder(h)                 # (n, 5, 32)  — tokens attend to each other
        return self.head(h.mean(dim=1))     # (n, 1)      — pool, then predict

model = PrimeTransformer()

# Training =========================================================================

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)

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

score("transformer", pred.numpy(), y_test)
