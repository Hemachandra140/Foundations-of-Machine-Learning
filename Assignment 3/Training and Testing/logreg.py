import numpy as np
import pandas as pd
import re
from collections import Counter
import pickle

D = pd.read_csv("train.csv")
np.random.seed(14)

X = D['email']
y = D['label']

word_count = Counter()

for text in X:
    words = set(re.findall(r'\b\w+\b', text.lower()))
    word_count.update(words)

Corpus = sorted([word for word, count in word_count.items() if count > 10])

word_index = {word: i for i, word in enumerate(Corpus)}

d = len(Corpus)

print("Vocabulary size:", d)

rng = np.random.default_rng(42)
w = rng.normal(0, 0.01, size=d)

eta = 0.1
epochs = 1000

# Preprocess emails once
emails = []

for text in X:
    words = set(re.findall(r'\b\w+\b', text.lower()))
    indices = [word_index[word] for word in words if word in word_index]
    emails.append(indices)

Y = y.to_numpy()

for epoch in range(epochs):

    gradient = np.zeros(d)
    errors = 0

    for i in range(n):

        z = 0
        for j in emails[i]:
            z += w[j]

        # Sigmoid
        if z >= 0:
            p_hat = 1 / (1 + np.exp(-z))
        else:
            ez = np.exp(z)
            p_hat = ez / (1 + ez)

        error = Y[i] - p_hat

        for j in emails[i]:
            gradient[j] += error

        # 0-1 classification error
        if (z >= 0) != Y[i]:
            errors += 1

    # Gradient ascent
    w = w + eta * gradient / n

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: 0-1 Error = {errors/n:.4f}")
T = pd.read_csv("test.csv")

X_test = T['email']
y_test = T['label']

C = np.zeros((2, 2), dtype=int)

for i in range(len(X_test)):
    text = X_test[i].lower()
    words = set(re.findall(r'\b\w+\b', text))

    # Compute w^T x
    z = 0
    for word in words:
        if word in word_index:
            z += w[word_index[word]]

    # Predict class
    yp = 1 if z >= 0 else 0

    C[int(y_test[i]), yp] += 1

print("Confusion Matrix")
print(C)

print("Accuracy:")
print((C[0,0] + C[1,1]) * 100 / len(X_test))

np.save("w_logistic.npy", w)
with open("word_index.pkl", "wb") as f:
    pickle.dump(word_index, f)