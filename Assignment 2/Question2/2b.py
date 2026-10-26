import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

X = pd.read_csv("Dataset2Train.csv", header=None)
X = X.to_numpy()

np.random.seed(19)

Y = X[:, -1]
X = X[:, :-1]

m = X.shape[0]
n = X.shape[1]

X=X.T

def gradient(X,Y,w):
    return 2*(X@X.T@w-X@Y)

def linear_regression_gd(X,Y,m,n):
    w = np.random.uniform(0,1,size=n)
    step = 1/(2*np.linalg.norm(X@X.T))
    w_ml = np.load("w_ml.npy")
    itr_err=[]

    for i in tqdm(range(10000), desc="Iterations"):

        g = gradient(X,Y,w)
        if np.linalg.norm(g)<1e-1:
            return w,itr_err
        w = w - step*g
        
        e = w-w_ml
        e = np.dot(e,e)
        itr_err.append(np.sqrt(e))
        
    return w,itr_err

w,e = linear_regression_gd(X,Y,m,n)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(e) + 1), e)
plt.xlabel("Iterations")
plt.ylabel(r"$\|w_i-w_{\mathrm{ML}}\|_2^2$")
plt.title("Error vs Iterations")
plt.show()

print("The least squares estimator obtained by gradient descent is :w_GD=")
print(w)
print(f"The error between w_ML and w_GD is: {e[-1]}")