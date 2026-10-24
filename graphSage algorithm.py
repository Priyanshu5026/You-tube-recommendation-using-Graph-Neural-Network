# =======================
# IMPORTS
# =======================
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import random

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)

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
X = df[['views','likes','dislikes','comment_count']].values

# =======================
# LABEL
# =======================
score = (
    0.6*np.log1p(X[:,0]) +
    0.7*np.sqrt(X[:,1]) -
    0.25*X[:,2] +
    0.5*np.log1p(X[:,3])
)
score += np.random.normal(0,0.1,size=score.shape)
y = (score > np.percentile(score,55)).astype(int)

# =======================
# NORMALIZATION
# =======================
X = StandardScaler().fit_transform(X)

# =======================
# GRAPH
# =======================
k = 10
# best balance accuracy + speed
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
_, indices = nbrs.kneighbors(X)

edges = []
for i in range(len(X)):
    for j in indices[i]:
        if i != j:
            edges.append([i,j])

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

train_idx, temp_idx = train_test_split(
    idx, test_size=0.2, stratify=y, random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=42
)

train_idx = torch.tensor(train_idx)
val_idx = torch.tensor(val_idx)
test_idx = torch.tensor(test_idx)

# =======================
# MODEL
# =======================
class GraphSAGEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(X.shape[1], 128)
        self.conv2 = SAGEConv(128, 64)
        self.conv3 = SAGEConv(64, 32)
        self.lin = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = self.lin(x)
        return x.squeeze()

model = GraphSAGEModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)

# =======================
# TRAINING
# =======================
train_losses, val_losses = [], []
acc_list, precision_list, recall_list, f1_list = [], [], [], []

best_loss = float('inf')
patience = 10
counter = 0

for epoch in range(120):

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
        val_loss = F.binary_cross_entropy_with_logits(val_out[val_idx], y[val_idx])
        val_losses.append(val_loss.item())

        probs = torch.sigmoid(val_out[val_idx]).cpu().numpy()
        y_val = y[val_idx].cpu().numpy()
        pred = (probs > 0.5).astype(int)

        acc_list.append(accuracy_score(y_val, pred))
        precision_list.append(precision_score(y_val, pred))
        recall_list.append(recall_score(y_val, pred))
        f1_list.append(f1_score(y_val, pred))

    # EARLY STOPPING
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping at epoch:", epoch)
            break

model.load_state_dict(best_model)

# =======================
# THRESHOLD TUNING
# =======================
with torch.no_grad():
    val_probs = torch.sigmoid(model(data)[val_idx]).cpu().numpy()
    y_val = y[val_idx].cpu().numpy()


# =======================
# TEST
# =======================
with torch.no_grad():
    probs = torch.sigmoid(model(data)[test_idx]).cpu().numpy()
    y_true = y[test_idx].cpu().numpy()

y_pred = (probs > 0.55).astype(int)

print("\n===== FINAL PERFORMANCE =====")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))

# =======================
# CONFUSION MATRIX
# =======================
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# =======================
# SMOOTH FUNCTION
# =======================
def smooth(data, weight=0.85):
    sm = []
    last = data[0]
    for point in data:
        val = last*weight + (1-weight)*point
        sm.append(val)
        last = val
    return sm

# =======================
# LOSS CURVE
# =======================
plt.plot(smooth(train_losses), label="Train")
plt.plot(smooth(val_losses), label="Val")
plt.xlabel("Epoches")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()

# =======================
# METRIC CURVES (SEPARATE)
# =======================
plt.plot(smooth(acc_list))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.show()

plt.plot(smooth(precision_list))
plt.xlabel("Epoches")
plt.ylabel("Precision")
plt.title("Precision Curve")
plt.show()

# =======================
# ROC
# =======================
fpr, tpr, _ = roc_curve(y_true, probs)
auc = roc_auc_score(y_true, probs)

plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve")
plt.show()

print("AUC:", auc)

# =======================
# PIE CHART
# =======================
sizes = [len(train_idx), len(val_idx), len(test_idx)]
labels = ['Train (80%)','Val (10%)','Test (10%)']
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Dataset Split")
plt.show()