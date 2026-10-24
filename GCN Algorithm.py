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
from torch_geometric.nn import GCNConv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.neighbors import NearestNeighbors

# =======================
# SEED
# =======================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# =======================
# LOAD DATA
# =======================
df = pd.read_csv("GBvideos.csv")

# =======================
# SAFE FEATURES
# =======================
features = df[['views', 'likes', 'dislikes', 'comment_count']]
X = features.values

# =======================
# LABEL
# =======================
score = (
    0.5 * X[:, 0] +
    0.4 * X[:, 1] -
    0.3 * X[:, 2] +
    0.4 * X[:, 3]
)

y = (score > np.percentile(score, 60)).astype(int)

# =======================
# NORMALIZATION
# =======================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =======================
# KNN GRAPH
# =======================
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
_, indices = nbrs.kneighbors(X)

edges = []
for i in range(len(X)):
    for j in indices[i]:
        if i != j:
            edges.append([i, j])

edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)

# =======================
# TENSOR
# =======================
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

data = Data(x=X, edge_index=edge_index, y=y)

# =======================
# SPLIT (80-10-10)
# =======================
idx = np.arange(len(y))
train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=seed)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)

train_idx = torch.tensor(train_idx)
val_idx = torch.tensor(val_idx)
test_idx = torch.tensor(test_idx)

# =======================
# MODEL (GCN)
# =======================
class GCNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(X.shape[1], 32)
        self.conv2 = GCNConv(32, 16)
        self.lin = torch.nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)

        return x.squeeze()

model = GCNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# =======================
# TRAINING
# =======================
precision_list = [0]
recall_list = [0]
f1_list = [0]
acc_list = [0]

train_losses, val_losses = [], []

for epoch in range(60):

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

        probs = torch.sigmoid(val_out[val_idx]).cpu().numpy()
        y_true_val = y[val_idx].cpu().numpy()
        y_pred_val = (probs > 0.5).astype(int)

        precision_list.append(precision_score(y_true_val, y_pred_val, zero_division=0))
        recall_list.append(recall_score(y_true_val, y_pred_val, zero_division=0))
        f1_list.append(f1_score(y_true_val, y_pred_val, zero_division=0))
        acc_list.append(accuracy_score(y_true_val, y_pred_val))

        val_loss = F.binary_cross_entropy_with_logits(val_out[val_idx], y[val_idx])
        val_losses.append(val_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

# =======================
# TEST
# =======================
model.eval()
with torch.no_grad():
    out = model(data)
    probs = torch.sigmoid(out[test_idx]).cpu().numpy()
    y_true = y[test_idx].cpu().numpy()

y_pred = (probs > 0.4).astype(int)

# =======================
# FINAL METRICS
# =======================
print("\n===== FINAL PERFORMANCE =====")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))

# =======================
# CONFUSION MATRIX
# =======================
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# =======================
# SMOOTH FUNCTIONS
# =======================
def smooth(data, weight=0.6):
    smoothed = []
    last = data[0]
    for point in data:
        val = last * weight + (1 - weight) * point
        smoothed.append(val)
        last = val
    return smoothed

# =======================
# CURVES
# =======================
plt.plot(smooth(acc_list))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.show()

plt.plot(smooth(precision_list))
plt.xlabel("epochs")
plt.ylabel("Precision")
plt.title("Precision Curve")
plt.show()

plt.plot(smooth(train_losses), label="Train Loss")
plt.plot(smooth(val_losses), label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()

fpr, tpr, _ = roc_curve(y_true, probs)
plt.plot(smooth(fpr), smooth(tpr))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

sizes = [len(train_idx), len(val_idx), len(test_idx)]
labels = ['Train (80%)', 'Validation (10%)', 'Test (10%)']

plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Dataset Split")
plt.show()