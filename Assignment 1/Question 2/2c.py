import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_excel("Dataset2.xlsx")
X = df.to_numpy()

X=X.T
n=X.shape[0]
m=X.shape[1]

np.random.seed(19)

def K_1(X,n,m,p):
    K = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            t=np.dot(X[:, i],X[:, j])+1
            K[i][j]=t**p
    I = np.ones((m,m))
    K = K-I@K/m - K@I/m + I@K@I/(m**2)
    return K

def K_l(X,n,m,s,k):
    K=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            x=X[:, i]
            y=X[:, j]
            d=np.linalg.norm(x-y)
            K[i][j]=np.exp(-(d)**2/(2*s**2))
    d = np.sum(K, axis=1)
    D_inv_sqrt = np.diag(1 / np.sqrt(d))
    L = np.eye(m) - D_inv_sqrt @ K @ D_inv_sqrt
    return L

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


def Spectral_Clustering(K,m,k):
    S,V = np.linalg.eigh(K)
    idx = np.argsort(S)[::-1]
    S = S[idx]
    V = V[:, idx]
    H = V[:, :k]

    for i in range(m):
        r = np.linalg.norm(H[i, :])
        for j in range(k):
            H[i][j]/=r

    return Lloyds(H.T,k,m,k)

def plot_clust(Z,X):

    c = ['red', 'blue', 'green', 'orange']
    
    colors = [c[v] for v in Z]
        
    plt.scatter(X[0, :],X[1, :],color=colors)
    legend_elements = [Line2D([0], [0], marker='o', color='w',label='Cluster 1', markerfacecolor='red', markersize=8),Line2D([0], [0], marker='o', color='w',label='Cluster 2', markerfacecolor='blue', markersize=8),Line2D([0], [0], marker='o', color='w',label='Cluster 3', markerfacecolor='green', markersize=8),Line2D([0], [0], marker='o', color='w',label='Cluster 4', markerfacecolor='orange', markersize=8)]
    
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend(handles=legend_elements, title="Clusters")
    plt.grid(True)
    plt.title("Spectral Clustering with Laplacian Matrix sigma=0.3")
    plt.show()

K = K_1(X,n,m,2)
Z,E,C=Spectral_Clustering(K,m,4)
plot_clust(Z,X)

L = K_l(X,n,m,0.3,4)
Z,E,C=Spectral_Clustering(-L,m,4)
plot_clust(Z,X)