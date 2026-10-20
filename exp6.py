import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and clean data
df = pd.read_excel("Online Retail.xlsx")
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# Group data by customer
c = df.groupby("CustomerID").agg(
    TotalSpend=("TotalAmount", "sum"),
    TotalQuantity=("Quantity", "sum"),
    Transactions=("InvoiceNo", "nunique"),
    Products=("StockCode", "nunique"),
    AvgOrder=("TotalAmount", "mean")
).reset_index()

# Create 3 customer groups
c["Segment"] = pd.qcut(c["TotalSpend"], 3,
                       labels=["Low", "Medium", "High"])

# Select features and target
features = ["TotalQuantity", "Transactions", "Products", "AvgOrder"]
X = c[features]
y = c["Segment"]

# Split data
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train ID3 decision tree
m = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
m.fit(Xtr, ytr)
p = m.predict(Xte)

# Results
print("Predictions:", p[:20])
print("Accuracy:", accuracy_score(yte, p))
print(classification_report(yte, p))
print("Confusion Matrix:\n", confusion_matrix(yte, p))

# Plot decision tree
plt.figure(figsize=(18, 8))
plot_tree(m, feature_names=features,
          class_names=["Low", "Medium", "High"],
          filled=True, rounded=True)
plt.show()

# Plot feature importance
plt.barh(features, m.feature_importances_)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()
