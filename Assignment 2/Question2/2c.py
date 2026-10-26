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

def rn_sample(X,Y,m,k):
    indices = np.random.choice(m, size=k, replace=False)
    X = X[:, indices]
    Y = Y[indices]

    return X,Y

def gradient(X,Y,w):
    return 2*(X@X.T@w-X@Y)

def linear_regression_sgd(X,Y,m,n,k):
    w = np.random.uniform(0,1,size=n)
    step = 2e-4
    w_ml = np.load("w_ml.npy")
    itr_err=[]

    w_sgd=np.zeros(n)
    for i in tqdm(range(100000), desc="Iterations"):

        Xs,Ys = rn_sample(X,Y,m,k)
        g = gradient(Xs,Ys,w)
        if np.linalg.norm(g)<1e-1:
            w_sgd/=(i+1)
            es = w_sgd-w_ml
            es = np.dot(es,es)
            return w,w_sgd,itr_err,es
        w = w - step*g

        w_sgd+=w
        e = w-w_ml
        e = np.dot(e,e)
        itr_err.append(np.sqrt(e))

    w_sgd/=100000
    es = w_sgd-w_ml
    es = np.dot(es,es)
    es = np.sqrt(es)
    return w,w_sgd,itr_err,es

w,w_sgd,e,es = linear_regression_sgd(X,Y,m,n,100)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(e) + 1), e)
plt.xlabel("Iterations")
plt.ylabel(r"$\|w_i-w_{\mathrm{ML}}\|_2^2$")
plt.title("Error vs Iterations")
plt.show()

print("The weight vector after the last update in the SGD algorithm is :w=")
print(w)
print("The least squares estimator obtained by stochastic gradient descent is :w_SGD=")
print(w_sgd)
print(f"The error between w_ML and w is: {e[-1]}")
print(f"The error between w_ML and w_sgd is: {es}")