
import numpy as np
import pickle
import re
import os

# Load trained weights
w = np.load("w_logistic.npy")

# Load word indices
with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)


def LogisticRegressionClassifier(text):

    words = set(re.findall(r'\b\w+\b', text.lower()))

    z = 0.0

    for word in words:
        if word in word_index:
            z += w[word_index[word]]

    if z >= 0:
        return 1
    else:
        return 0


def predict():

    folder = "test"

    files = sorted(
        f for f in os.listdir(folder)
        if f.endswith(".txt")
        and os.path.isfile(os.path.join(folder, f))
    )

    predictions = {}

    for filename in files:

        path = os.path.join(folder, filename)

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        label = LogisticRegressionClassifier(text)

        predictions[filename] = label

        print(f"{filename}: {label}")

    return predictions

predict()
