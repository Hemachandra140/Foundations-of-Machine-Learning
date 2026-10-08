import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


df = pd.read_excel("Dataset1.xlsx")
X = df.iloc[:, :2].to_numpy()
y = df.iloc[:, 2].to_numpy()

m=X.shape[0]
n=X.shape[1]

X = X.T

def K_1(X,n,m,p):
    T=X.T@X
    I = np.ones((m,m))
    T+=I
    T=T**p
    return T

def K_2(X,n,m,p):
    K = np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            delta = X[:,i]-X[:,j]
            K[i][j]=np.exp(-(np.dot(delta,delta))/(2*(p**2)))

    return K

def K_p(X,n,m,p,s):
    K = np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            k_1=(np.dot(X[:, i],X[:, j])+1)**p
            delta = X[:, i]-X[:, j]
            K[i][j]=k_1*np.exp(-(np.dot(delta,delta))/(2*(s**2)))

    return K

def K_s(X,n,m,p,s):
    K = np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            k_1=(np.dot(X[:, i],X[:, j])+1)**p
            delta = X[:, i]-X[:, j]
            K[i][j]=k_1+np.exp(-(np.dot(delta,delta))/(2*(s**2)))

    return K

def center_K(K,m):
    I = np.ones((m,m))
    K = K-I@K/m - K@I/m + I@K@I/(m**2)
    return K

def KPCA(K,m):
    S,V = np.linalg.eigh(K)
    
    idx = np.argsort(S)[::-1]
    S = S[idx]
    V = V[:, idx]
    
    tol = 1e-10
    mask = S > tol
    
    S = S[mask]
    V = V[:, mask]
    
    V=V/(np.sqrt(S))

    return S,V

colors = ["red" if yi == -1 else "blue" for yi in y]

def plot_proj(K,S,V,m,title):
    Z = K@V[:, :2]
    plt.scatter(Z[:, 0], Z[:, 1],c=colors)
    plt.xlabel("Projection value along 1st PC")
    plt.ylabel("Projection value along 2nd PC")
    legend_elements = [Line2D([0], [0], marker='o', color='w',label='-1', markerfacecolor='red', markersize=8),Line2D([0], [0], marker='o', color='w',label='1', markerfacecolor='blue', markersize=8)]
    plt.legend(handles=legend_elements, title="Class")
    plt.title(title)
    plt.show()

K = center_K(K_1(X,n,m,2), m)
S,V = KPCA(K,m)
plot_proj(K,S,V,m,"Kernel (x.y+1)^2")

K = center_K(K_1(X,n,m,5), m)
S,V = KPCA(K,m)
plot_proj(K,S,V,m,"Kernel (x.y+1)^5")

K = center_K(K_2(X,n,m,0.1), m)
S,V = KPCA(K,m)
plot_proj(K,S,V,m,"RBF Kernel sigma=0.1")

K = center_K(K_2(X,n,m,0.5), m)
S,V = KPCA(K,m)
plot_proj(K,S,V,m,"RBF Kernel sigma=0.5")

K = center_K(K_2(X,n,m,1), m)
S,V = KPCA(K,m)
plot_proj(K,S,V,m,"RBF Kernel sigma=1")

K = center_K(K_2(X,n,m,2), m)
S,V = KPCA(K,m)
plot_proj(K,S,V,m,"RBF Kernel sigma=2")

K = center_K(K_p(X,n,m,3,0.1), m)
S,V = KPCA(K,m)
plot_proj(K,S,V,m,"Polynomial kernel 3 * RBF Kernel sigma=0.1")

K = center_K(K_s(X,n,m,3,0.1), m)
S,V = KPCA(K,m)
plot_proj(K,S,V,m,"Polynomial kernel 3 + RBF Kernel sigma=0.1")