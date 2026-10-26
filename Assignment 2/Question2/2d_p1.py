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

lam = [1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5]

def gradient(X,Y,w,l):
    return (X@X.T@w-X@Y+l*w)

def create_partitions(X,Y,m,k):
    idx = np.random.permutation(m)
    X = X[:, idx]
    Y = Y[idx]
    
    indices = np.array_split(np.arange(m), k)

    X_folds = []
    Y_folds = []

    for fold in indices:
        X_folds.append(X[:, fold])
        Y_folds.append(Y[fold])

    return X_folds, Y_folds

def reg_gd(X,Y,n,l):
    w = np.random.uniform(0,1e-3,size=n)
    step = 1/(l+5e5)
    for i in range(100000):
        g = gradient(X,Y,w,l)
        if np.linalg.norm(g)<1e-1:
            return w
        w = w - step*g
        
    return w

def test_w(X,Y,w):
    e = (X.T)@w-Y
    return np.dot(e,e)
    
def ridge(X,Y,m,n,lam,k):
    X_folds, Y_folds = create_partitions(X,Y,m,k)
    Err = []
    for l in tqdm(lam, desc="Iterating over Lambda"):
        E=0
        for i in range(k):
            Xt = X_folds[i]
            Yt = Y_folds[i]
            X_train = np.concatenate([X_folds[j] for j in range(k) if j != i],axis=1)
            Y_train = np.concatenate([Y_folds[j] for j in range(k) if j != i])

            w = reg_gd(X_train,Y_train,n,l)
            e = test_w(Xt,Yt,w)
            E+=e
            
        Err.append(E/k)
    return Err

E = ridge(X,Y,m,n,lam,10)

plt.figure(figsize=(8, 5))
plt.plot(range(1,len(E)+1), E)
plt.xlabel("Lambda Index")
plt.ylabel("Average Error over the K-folds")
plt.title("Error vs Lambda")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(lam, E)
plt.xlabel("Lambda")
plt.ylabel("Average Error over the K-folds")
plt.title("Error vs Lambda")
plt.show()

i = np.argmin(E)
print(E)
print(f"Lambda for which the average error is minimum is: {lam[i]}")