import numpy as np
import pandas as pd

df = pd.read_excel("Dataset1.xlsx")
X = df.iloc[:, :2].to_numpy()
y = df.iloc[:, 2].to_numpy()

X=X.T

n=X.shape[0]
m=X.shape[1]

def mean(X,n,m):
    mu = np.zeros(n)
    for i in range(n):
        for j in range(m):
            mu[i]+=X[i][j]
    mu/=m
    return mu

def center_data(X,n,m):
    mu = mean(X,n,m)
    
    for i in range(n):
        for j in range(m):
            X[i][j] -= mu[i]
    return X

X = center_data(X,n,m)

def get_C(X,n,m):
    C=X@X.T
    return C/m

C = get_C(X,n,m)

S,V = np.linalg.eigh(C)

idx = np.argsort(S)[::-1]
S = S[idx]
V = V[:, idx]

l = len(S)
for i in range(l):
    print(f"{S[i]/(np.sum(S))*100}% of Variance in data is explained by the {i+1}th principle component")