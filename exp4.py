import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import *

# Load data
tr = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
te = fetch_20newsgroups(subset='test', remove=('headers','footers','quotes'))

# Convert text to numbers
v = CountVectorizer(stop_words='english', min_df=2)
Xtr = v.fit_transform(tr.data)
Xte = v.transform(te.data)

# Multinomial Naive Bayes
m = MultinomialNB()
m.fit(Xtr, tr.target)
p1 = m.predict(Xte)

# Bernoulli Naive Bayes
v2 = CountVectorizer(stop_words='english', min_df=2, binary=True)
A = v2.fit_transform(tr.data)
B = v2.transform(te.data)

b = BernoulliNB()
b.fit(A, tr.target)
p2 = b.predict(B)

# Compare models
results = []
for name, ypred in [('MultinomialNB', p1), ('BernoulliNB', p2)]:
    results.append([
        name, accuracy_score(te.target, ypred),
        precision_score(te.target, ypred, average='weighted'),
        recall_score(te.target, ypred, average='weighted'),
        f1_score(te.target, ypred, average='weighted')
    ])

r = pd.DataFrame(results, columns=['Model','Accuracy','Precision','Recall','F1'])
print(r)

# Bar graph
r.set_index('Model').plot(kind='bar')
plt.title("MNB vs BNB")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()
