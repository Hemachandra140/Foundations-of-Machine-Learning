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

L = K_l(X,n,m,0.3,4)

S,V = np.linalg.eigh(-L)
idx = np.argsort(S)[::-1]
S = S[idx]
V = V[:, idx]
H = V[:, :4]

Z = [np.argmax(H[i, :]) for i in range(m)]

c = ['red', 'blue', 'green', 'orange']
    
colors = [c[v] for v in Z]
    
plt.scatter(X[0, :],X[1, :],color=colors)
legend_elements = [Line2D([0], [0], marker='o', color='w',label='Cluster 1', markerfacecolor='red', markersize=8),Line2D([0], [0], marker='o', color='w',label='Cluster 2', markerfacecolor='blue', markersize=8),Line2D([0], [0], marker='o', color='w',label='Cluster 3', markerfacecolor='green', markersize=8),Line2D([0], [0], marker='o', color='w',label='Cluster 4', markerfacecolor='orange', markersize=8)]

plt.xlabel("x₁")
plt.ylabel("x₂")
plt.legend(handles=legend_elements, title="Clusters")
plt.grid(True)
plt.title("Custom Clustering with Laplacian Matrix sigma=0.3")
plt.show()