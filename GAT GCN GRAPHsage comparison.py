import matplotlib.pyplot as plt
import numpy as np

# =========================
# MODEL RESULTS
# =========================

models = ["GCN", "GraphSAGE", "GAT","Hybrid Series","Hybrid Parallel"]
accuracy = [0.9072, 0.9231, 0.9126,0.9054,0.9008]
precision = [0.9397, 0.8920, 0.9443,0.8561,0.8567]
recall = [0.8277, 0.9434, 0.8370,0.9177,0.9029]
f1_score = [0.8801, 0.9170, 0.8874,0.8858,0.8792]


# =========================
# BAR GRAPH FUNCTION
# =========================

def plot_metric(values, title):
    plt.figure()
    x = np.arange(len(models))
    plt.bar(x, values)

    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel("Score")

    # Value labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

    plt.show()


# =========================
# PLOT ALL METRICS
# =========================

plot_metric(accuracy, "Model Comparison - Accuracy")
plot_metric(precision, "Model Comparison - Precision")
plot_metric(recall,"Model Comparison - recall")
plot_metric(f1_score, "Model Comparison - f1_score")


# =========================
# LINE GRAPH
# =========================

plt.figure()

plt.plot(models, accuracy, marker='o', label='Accuracy')
plt.plot(models, precision, marker='o', label='Precision')
plt.plot(models, recall, marker='o', label='Recall')
plt.plot(models, f1_score, marker='o', label='F1 Score')

plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Performance Comparison Curve")
plt.legend()
plt.grid()

# =========================
# COMBINED GRAPH
# =========================

x = np.arange(len(models))
width = 0.15

plt.figure()

plt.bar(x - 2 * width, accuracy, width, label='Accuracy')
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x + width, recall, width, label='recall')
plt.bar(x + 2 * width, f1_score, width, label='f1_score')

plt.xticks(x, models)
plt.ylim(0, 1)
plt.title("Overall Model Comparison")
plt.legend()
plt.show()


