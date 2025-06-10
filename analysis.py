import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load CSV
df = pd.read_csv("results/results_log.csv")

# Initialize true and predicted labels
y_true = []
y_pred = []

# Assume ground truth is "Test1" (criminal) vs "Unknown" (not criminal)
# Mark all predictions as initially correct
for i, row in df.iterrows():
    if row['result'] == 'Match':
        y_pred.append('Test1')
        y_true.append('Test1')  # Assume correct initially
    elif row['result'] == 'Unknown':
        y_pred.append('Unknown')
        y_true.append('Unknown')  # Assume correct initially

# Fix known mislabeled cases
# Last 4 matches were wrong (false positives)
for i in range(len(y_true) - 4, len(y_true)):
    y_true[i] = 'Unknown'  # They were actually not matches

# Adjust 7 unknowns to be false negatives (they should have been matched)
# Since we don't know which, we’ll mark the first 7 "Unknown" predictions as missed matches
false_negatives_fixed = 0
for i in range(len(y_pred)):
    if y_pred[i] == 'Unknown' and y_true[i] == 'Unknown':
        y_true[i] = 'Test1'  # Actually a missed match
        false_negatives_fixed += 1
        if false_negatives_fixed == 7:
            break

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=["Test1", "Unknown"])
labels = ["Criminal (Test1)", "Not Criminal (Unknown)"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels))
