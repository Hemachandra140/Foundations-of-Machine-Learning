import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

X = pd.read_csv("Dataset1.csv", header=None)
X = X.to_numpy()

m = X.shape[0]
n = X.shape[1]

np.random.seed(19)


def initialize(K,m,n):
    T = np.random.rand(K)
    T /=np.sum(T)

    P = np.random.uniform(0.1, 0.9, size=((K,n)))

    return T,P

def EM(X,m,n,K):

    T,P = initialize(K,m,n)
    L = np.zeros((m,K))
    T1 = np.zeros(K)
    P1 = np.zeros((K,n))

    while True:

        #E step
        for i in range(m):
            te=0
            for j in range(K):
                e_ij=1
                for k in range(n):
                    e_ij*=(P[j][k])**(X[i][k])*(1-P[j][k])**(1-X[i][k])
                te += T[j]*e_ij
                L[i][j] = T[j]*e_ij
            for j in range(K):
                L[i][j] /= te

        #M step
        for j in range(K):
            for k in range(n):
                s1=0
                s2=0
                for i in range(m):
                    s1+=L[i][j]*X[i][k]
                    s2+=L[i][j]

                P1[j][k]=s1/s2
                T1[j]=s2/m

        D1=P1-P
        D2=T1-T
        if np.linalg.norm(D1)+np.linalg.norm(D2)<1e-5:
            return T,P
        
        P=P1.copy()
        T=T1.copy()

def get_log_likelihood(X,T,P,n,m,K):
    logl=0
    for i in range(m):
        t2=0
        for j in range(K):
            t1=1
            for k in range(n):
                t1*=(P[j][k])**(X[i][k])*(1-P[j][k])**(1-X[i][k])
            t2+=T[j]*t1
        logl+=np.log(t2)

    return logl

avg_logl=0
for i in tqdm(range(5), desc="EM runs"):
    T, P = EM(X, m, n, 4)
    avg_logl+= get_log_likelihood(X,T,P,n,m,4)

print(f"The average of the log-likelihood using EM for 5 random initialization is {avg_logl/5}")