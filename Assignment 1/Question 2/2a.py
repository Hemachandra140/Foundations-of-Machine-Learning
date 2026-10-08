import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Dataset2.xlsx")
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
    error_per_iter=[]
    ep=0
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

        e=0
        for i in range(m):
            t = [np.dot(X[:, i]-centers[j],X[:, i]-centers[j]) for j in range(k)]
            Z[i] = np.argmin(t)
            e+= float((t[Z[i]]))

        if Z_t==Z:
            return Z,error_per_iter,centers
        
        error_per_iter.append(e)
        ep=e

def plot_clust(Z,X,C):

    c = ['red', 'blue', 'green', 'orange']
    
    colors = [c[v] for v in Z]
        
    plt.scatter(X[0, :],X[1, :],color=colors)

    for i, v in enumerate(C):
        plt.scatter(v[0],v[1],marker='X',s=200,color=c[i],edgecolors='black',linewidths=1.5,label=f'Center {i+1}')
    
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_err(E):
    plt.plot(range(1, len(E) + 1), E, marker='o')
    
    plt.xlabel("Iteration")
    plt.ylabel("K-means Error")
    plt.title("Error vs Iteration")
    plt.grid(True)
    plt.show()

for i in range(5):
    Z,E,C = Lloyds(X,n,m,4)
    plot_clust(Z,X,C)
    plot_err(E)