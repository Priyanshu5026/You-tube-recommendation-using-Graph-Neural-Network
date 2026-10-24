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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

# =======================
# DEVICE
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# LABEL CREATION
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

data = Data(x=X, edge_index=edge_index, y=y).to(device)

# =======================
# SPLIT
# =======================
idx = np.arange(num_nodes)

train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=seed)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)

train_idx = torch.tensor(train_idx).to(device)
val_idx = torch.tensor(val_idx).to(device)
test_idx = torch.tensor(test_idx).to(device)

# =======================
# MODEL
# =======================
class HybridGNN(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.gcn1 = GCNConv(in_features, 64)
        self.gcn2 = GCNConv(64, 32)

        self.gat1 = GATConv(in_features, 32, heads=2, concat=True)
        self.gat2 = GATConv(64, 32, heads=1, concat=True)

        self.lin = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index):

        x1 = F.relu(self.gcn1(x, edge_index))
        x1 = F.relu(self.gcn2(x1, edge_index))

        x2 = F.elu(self.gat1(x, edge_index))
        x2 = F.elu(self.gat2(x2, edge_index))

        x = torch.cat([x1, x2], dim=1)

        return self.lin(x).squeeze()

model = HybridGNN(X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# =======================
# METRICS
# =======================
train_losses = []
acc_list = []
prec_list = []
rec_list = []
f1_list = []

# =======================
# TRAINING
# =======================
for epoch in range(50):

    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    loss = F.binary_cross_entropy_with_logits(
        out[train_idx],
        data.y[train_idx]
    )

    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # =======================
    # EVALUATION
    # =======================
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.sigmoid(out).cpu().numpy()

    y_true = data.y.cpu().numpy()
    y_pred = (probs > 0.5).astype(int)

    acc_list.append(accuracy_score(y_true, y_pred))
    prec_list.append(precision_score(y_true, y_pred, zero_division=0))
    rec_list.append(recall_score(y_true, y_pred, zero_division=0))
    f1_list.append(f1_score(y_true, y_pred, zero_division=0))

    print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {acc_list[-1]:.4f}")

# =======================
# SMOOTH FUNCTIONS
# =======================
def smooth_curve(values, alpha=0.9):
    smoothed = []
    last = 0
    for v in values:
        last = alpha * last + (1 - alpha) * v
        smoothed.append(last)
    return smoothed

def normalize_start_zero(curve):
    curve = np.array(curve)
    curve = curve - curve[0]
    curve = curve / (curve.max() + 1e-8)
    return curve

# =======================
# FINAL RESULTS
# =======================
print("\n===== FINAL RESULTS =====")
print("Accuracy:", acc_list[-1])
print("Precision:", prec_list[-1])
print("Recall:", rec_list[-1])
print("F1:", f1_list[-1])

# =======================
# LOSS CURVE
# =======================
plt.figure()
plt.plot(normalize_start_zero(train_losses))
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# =======================
# ACCURACY CURVE
# =======================
plt.figure()
plt.plot(normalize_start_zero(smooth_curve(acc_list)))
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# =======================
import numpy as np
import matplotlib.pyplot as plt

# =======================
# SMOOTH FUNCTION
# =======================
def smooth(values, alpha=0.9):
    smoothed = []
    last = 0
    for v in values:
        last = alpha * last + (1 - alpha) * v
        smoothed.append(last)
    return smoothed

# =======================
# NORMALIZE
# =======================
def normalize(curve):
    curve = np.array(curve)
    curve = curve - curve[0]
    curve = curve / (curve.max() + 1e-8)
    return curve

# =======================
# PROCESSED METRICS
# =======================
precision = normalize(smooth(prec_list))
recall    = normalize(smooth(rec_list))
f1        = normalize(smooth(f1_list))

# =======================
# 1. PRECISION PLOT
# =======================
plt.figure()
plt.plot(precision, color='blue')
plt.title("Precision Curve")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.grid()
plt.show()

# =======================
# 2. RECALL PLOT
# =======================
plt.figure()
plt.plot(recall, color='green')
plt.title("Recall Curve")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.ylim(0, 1)
plt.grid()
plt.show()

# =======================
# 3. F1 SCORE PLOT
# =======================
plt.figure()
plt.plot(f1, color='red')
plt.title("F1 Score Curve")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.grid()
plt.show()

# =======================
# CONFUSION MATRIX
# =======================
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# =======================
# ROC CURVE
# =======================
fpr, tpr, _ = roc_curve(y_true, probs)

plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# =======================
# PIE CHART
# =======================
plt.figure()
plt.pie(
    [len(train_idx), len(val_idx), len(test_idx)],
    labels=["Train", "Val", "Test"],
    autopct="%1.1f%%"
)
plt.title("Dataset Split")
plt.show()