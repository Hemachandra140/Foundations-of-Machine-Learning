import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset1.csv", header=None)
X = df.to_numpy()

np.random.seed(19)

X=X.T
n=X.shape[0]
m=X.shape[1]

def get_uni_rn_initialization(m,k):
    Z = [np.random.randint(0,k) for i in range(m)]
    return Z

def Lloyds(X,n,m,k):
    Z = get_uni_rn_initialization(m,k)
    while True:
        Z_t = Z.copy()
        centers = [np.zeros(n) for i in range(k)]
        count = [0 for i in range(k)]
        for i,z in enumerate(Z):
            centers[z]+=X[:, i]
            count[z]+=1
        for i,v in enumerate(centers):
            if count[i]!=0:
                centers[i]=v/count[i]

        if Z_t==Z:
            return Z,centers

def get_parameters(Z,C,n,m,K):
    T = np.zeros(K)
    P = np.zeros((K,n))
    
    for j in range(K):
        T[j]=Z.count(j)
        T[j]/=len(Z)

    for j,v in enumerate(C):
        for k,u in enumerate(v):
            P[j][k]=u

    return T,P

def get_log_likelihood(X,T,P,n,m,K):
    logl=0
    for i in range(m):
        t2=0
        for j in range(K):
            t1=1
            for k in range(n):
                t1*=(P[j][k])**(X[k][i])*(1-P[j][k])**(1-X[k][i])
            t2+=T[j]*t1
        logl+=np.log(t2)

    return logl

avg_logl=0
for i in range(5):
    Z,C = Lloyds(X,n,m,4)
    T,P = get_parameters(Z,C,n,m,4)
    avg_logl+= get_log_likelihood(X,T,P,n,m,4)

print(f"The average of the log-likelihood using K-means for 5 random initialization is {avg_logl/5}")