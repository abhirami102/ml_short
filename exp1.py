import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Data
d = fetch_california_housing()
X = d.data[:, 2:3]
y = d.target

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

s = StandardScaler()
Xtr = s.fit_transform(Xtr)
Xte = s.transform(Xte)

# Gradient Descent
w, b = 0, 0
lr = 0.01
n = len(Xtr)
cost = []

for i in range(1000):
    p = w * Xtr.ravel() + b
    cost.append(np.mean((p-ytr)**2)/2)
    w -= lr * np.mean((p-ytr)*Xtr.ravel())
    b -= lr * np.mean(p-ytr)

p1 = w * Xte.ravel() + b
print("Gradient Descent")
print("Weight:", w, "Bias:", b)
print("MSE:", mean_squared_error(yte, p1))
print("R2:", r2_score(yte, p1))

# Normal Equation
A = np.c_[np.ones(len(Xtr)), Xtr]
B = np.c_[np.ones(len(Xte)), Xte]
t = np.linalg.inv(A.T @ A) @ A.T @ ytr
p2 = B @ t

print("\nNormal Equation")
print("Intercept:", t[0], "Slope:", t[1])
print("MSE:", mean_squared_error(yte, p2))
print("R2:", r2_score(yte, p2))

# Cost graph
plt.plot(cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Graph")
plt.show()

# Regression graph
i = np.argsort(Xte.ravel())
plt.scatter(Xte, yte)
plt.plot(Xte.ravel()[i], p2[i], 'r')
plt.xlabel("Average Rooms (scaled)")
plt.ylabel("House Price")
plt.title("Linear Regression")
plt.show()
