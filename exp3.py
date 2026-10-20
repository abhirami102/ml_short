import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

# Load data
d = load_breast_cancer()
X, y = d.data, d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale data
s = StandardScaler()
Xtr, Xte = s.fit_transform(Xtr), s.transform(Xte)

# Train models
models = [
    ("MLE", LogisticRegression(penalty=None, max_iter=5000)),
    ("MAP L2", LogisticRegression(penalty='l2', max_iter=5000)),
    ("MAP L1", LogisticRegression(penalty='l1', solver='liblinear', max_iter=5000))
]

results = []

for name, m in models:
    m.fit(Xtr, ytr)
    p = m.predict(Xte)

    print("\n", name)
    print("Accuracy:", accuracy_score(yte, p))
    print("Precision:", precision_score(yte, p))
    print("Recall:", recall_score(yte, p))
    print("F1:", f1_score(yte, p))
    print("Confusion Matrix:\n", confusion_matrix(yte, p))
    print("Zero coefficients:", np.sum(m.coef_[0] == 0))

    results.append([name, accuracy_score(yte, p),
                    precision_score(yte, p), recall_score(yte, p),
                    f1_score(yte, p)])

# Results table and bar graph
r = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
print(r)
r.set_index('Model').plot(kind='bar')
plt.title("Model Comparison")
plt.ylim(0.85, 1.02)
plt.tight_layout()
plt.show()

# ROC graph
for name, m in models:
    prob = m.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, prob)
    plt.plot(fpr, tpr, label=name + " (AUC=" + str(round(auc(fpr, tpr), 3)) + ")")

plt.plot([0, 1], [0, 1], '--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()
