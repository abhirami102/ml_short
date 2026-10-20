import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
d = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X, y = d.data, d.target.astype(int)

# Select 12000 images
X, _, y, _ = train_test_split(X, y, train_size=12000, stratify=y, random_state=42)

# Split into training and testing
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=2000, stratify=y, random_state=42)

# Scale pixels
Xtr, Xte = Xtr/255, Xte/255

# Test different K values
results = {}
for k in [1, 3, 5, 7, 9, 15]:
    m = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    m.fit(Xtr, ytr)

    start = time.time()
    p = m.predict(Xte)
    t = time.time() - start

    acc = accuracy_score(yte, p)
    results[k] = (acc, p)
    print("K:", k, "Accuracy:", round(acc*100, 2), "%", "Time:", round(t, 2), "s")

# Best K
best = max(results, key=lambda k: results[k][0])
print("Best K:", best)
print("Accuracy:", round(results[best][0]*100, 2), "%")

# Classification report
names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(classification_report(yte, results[best][1], target_names=names))
