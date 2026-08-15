# extrapolation/mlp.py
import torch
import torch.nn as nn
from steps_data import X_train, y_train, nx, ny, dy, evaluate

torch.manual_seed(0)

# Tensors (normalized with TRAIN statistics) ===============================
X_train_t = torch.tensor(nx(X_train), dtype=torch.float32).reshape(-1, 1)
y_train_t = torch.tensor(ny(y_train), dtype=torch.float32).reshape(-1, 1)

# Network =================================================================
ACT = nn.Tanh          # <- the only line that changes for the tanh run

model = nn.Sequential(
    nn.Linear(1, 64),
    ACT(),
    nn.Linear(64, 64),
    ACT(),
    nn.Linear(64, 1)
)

# Training ================================================================
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()
    if epoch % 400 == 0:
        print(f"epoch {epoch:5d}   train {loss.item():.6f}")

# Bridge back to raw units ================================================
# evaluate() hands you RAW seconds and expects RAW steps back.
def predict(x_raw):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(nx(x_raw), dtype=torch.float32).reshape(-1, 1))
    return dy(out.numpy().ravel())

evaluate(f"MLP ({ACT.__name__})", predict)
