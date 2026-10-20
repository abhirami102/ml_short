import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")
df = df[['displacement', 'mpg']].dropna()
X, y = df[['displacement']], df['mpg']

# Plot actual data
plt.scatter(X, y)
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.show()

# Split data
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
m = LinearRegression().fit(Xtr, ytr)
p = m.predict(Xte)
print("Linear MSE:", mean_squared_error(yte, p))
print("Linear R2:", r2_score(yte, p))

results = [(1, mean_squared_error(yte, p), r2_score(yte, p))]
models = {}

# Polynomial Regression
for d in range(2, 6):
    poly = PolynomialFeatures(degree=d)
    A, B = poly.fit_transform(Xtr), poly.transform(Xte)
    model = LinearRegression().fit(A, ytr)
    pred = model.predict(B)
    results.append((d, mean_squared_error(yte, pred), r2_score(yte, pred)))
    models[d] = (poly, model)

# Results table
r = pd.DataFrame(results, columns=['Degree', 'MSE', 'R2'])
print(r)

# Compare R2
plt.plot(r['Degree'], r['R2'], marker='o')
plt.xlabel("Degree")
plt.ylabel("R2")
plt.title("R2 Comparison")
plt.grid()
plt.show()

# Compare MSE
plt.bar(r['Degree'].astype(str), r['MSE'])
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.title("MSE Comparison")
plt.show()

# Plot all regression lines
xp = pd.DataFrame({'displacement': np.linspace(X.displacement.min(), X.displacement.max(), 300)})
plt.scatter(X, y, label="Actual")

plt.plot(xp, m.predict(xp), label="Linear")

for d, (poly, model) in models.items():
    plt.plot(xp, model.predict(poly.transform(xp)), label=f"Degree {d}")

plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.legend()
plt.grid()
plt.show()
