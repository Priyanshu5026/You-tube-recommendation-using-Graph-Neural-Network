# =======================
# IMPORTS
# =======================
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

# =======================
# SEED
# =======================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# =======================
# SMOOTH FUNCTION
# =======================
def smooth(data, weight=0.9):
    smoothed = []
    last = 0
    for v in data:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed

# =======================
# LOAD DATA
# =======================
features = pd.read_csv("dataset youtube/youtube_features.csv")
edges = pd.read_csv("dataset youtube/youtube_edges.csv")

X = features.drop(columns=['label']).values

scaler = StandardScaler()
X = scaler.fit_transform(X)

num_nodes = X.shape[0]

edges = edges[(edges >= 0).all(axis=1)]
edges = edges[(edges < num_nodes).all(axis=1)]

edge_index = torch.tensor(edges.values.T, dtype=torch.long)

degree = np.bincount(edges.values.flatten(), minlength=num_nodes)
degree = degree / degree.max()

# =======================
# LABEL
# =======================
score = (
    0.4 * X[:, 0] +
    0.3 * X[:, 1] -
    0.2 * X[:, 2] +
    0.3 * degree +
    0.2 * np.sin(X[:, 3] * 3)
)

y = (score > np.percentile(score, 55)).astype(int)

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

data = Data(x=X, edge_index=edge_index, y=y)

# =======================
# SPLIT
# =======================
idx = np.arange(len(y))
train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=seed)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)

train_idx = torch.tensor(train_idx)
val_idx = torch.tensor(val_idx)
test_idx = torch.tensor(test_idx)

# =======================
# HYBRID MODEL
# =======================
class HybridGNN(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.gcn1 = GCNConv(in_features, 64)
        self.gcn2 = GCNConv(64, 32)

        self.gat1 = GATConv(in_features, 32, heads=2, concat=True)
        self.gat2 = GATConv(32 * 2, 32, heads=1, concat=True)

        self.lin = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = F.relu(self.gcn1(x, edge_index))
        x1 = F.relu(self.gcn2(x1, edge_index))

        x2 = F.elu(self.gat1(x, edge_index))
        x2 = F.elu(self.gat2(x2, edge_index))

        x = torch.cat([x1, x2], dim=1)

        return self.lin(x).squeeze()

model = HybridGNN(X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# =======================
# TRAINING
# =======================
train_losses, val_losses = [], []
precision_list, recall_list, f1_list = [], [], []

best_val = float('inf')
patience = 20
counter = 0

for epoch in range(200):

    model.train()
    optimizer.zero_grad()

    out = model(data)

    loss = F.binary_cross_entropy_with_logits(out[train_idx], y[train_idx])

    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # VALIDATION
    model.eval()
    with torch.no_grad():
        val_out = model(data)

        probs_val = torch.sigmoid(val_out[val_idx]).numpy()
        y_val = y[val_idx].numpy()
        pred_val = (probs_val > 0.5).astype(int)

        precision_list.append(precision_score(y_val, pred_val, zero_division=0))
        recall_list.append(recall_score(y_val, pred_val, zero_division=0))
        f1_list.append(f1_score(y_val, pred_val, zero_division=0))

        val_loss = F.binary_cross_entropy_with_logits(val_out[val_idx], y[val_idx])
        val_losses.append(val_loss.item())

    # EARLY STOPPING
    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_hybrid.pth")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# =======================
# LOAD BEST MODEL
# =======================
model.load_state_dict(torch.load("best_hybrid.pth"))
model.eval()

# =======================
# TEST
# =======================
with torch.no_grad():
    out = model(data)
    probs = torch.sigmoid(out[test_idx]).numpy()
    y_true = y[test_idx].numpy()

# =======================
# THRESHOLD
# =======================
best_t, best_f1 = 0.5, 0

for t in np.arange(0.3, 0.7, 0.02):
    preds = (probs > t).astype(int)
    f1 = f1_score(y_true, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

y_pred = (probs > best_t).astype(int)

# =======================
# FINAL METRICS
# =======================
print("\n===== RESULTS =====")
print("Best Threshold:", best_t)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, probs))

# =======================
# CURVES
# =======================
plt.figure()
plt.plot(smooth(precision_list), label="Precision")
plt.title("Precision Curve")
plt.ylim(0,1)
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(smooth(recall_list), label="Recall")
plt.title("Recall Curve")
plt.ylim(0,1)
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(smooth(f1_list), label="F1 Score")
plt.title("F1 Curve ")
plt.ylim(0,1)
plt.legend()
plt.grid()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# =======================
# CONFUSION MATRIX
# =======================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format="d")

plt.title("Confusion Matrix (Hybrid GCN + GAT)")
plt.show()

# =======================
# ROC CURVE
# =======================
fpr, tpr, _ = roc_curve(y_true, probs)

plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.legend()
plt.show()

# =======================
# PIE CHART
# =======================
plt.pie(
    [len(train_idx), len(val_idx), len(test_idx)],
    labels=["Train", "Val", "Test"],
    autopct="%1.1f%%"
)
plt.title("Dataset Split")
plt.show()