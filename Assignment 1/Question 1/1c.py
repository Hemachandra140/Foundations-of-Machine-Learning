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

def K_p(X,n,m,p,s):
    K = np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            k_1=(np.dot(X[:, i],X[:, j])+1)**p
            delta = X[:, i]-X[:, j]
            K[i][j]=k_1*np.exp(-(np.dot(delta,delta))/(2*(s**2)))

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

K = center_K(K_p(X,n,m,3,0.1), m)
S,V = KPCA(K,m)
Z = K@V[:, :2]
plt.scatter(Z[:, 0], Z[:, 1],c=colors)
plt.xlim(-3, 15)
plt.ylim(-5, 8)
x_line = np.linspace(-5, 15, 100)
y_line = -1 * x_line -5/2
plt.plot(x_line, y_line, 'k-', linewidth=2, label='y+x+2.5=0')
plt.xlabel("Projection value along 1st PC")
plt.ylabel("Projection value along 2nd PC")
legend_elements = [Line2D([0], [0], marker='o', color='w',label='-1', markerfacecolor='red', markersize=8),Line2D([0], [0], marker='o', color='w',label='1', markerfacecolor='blue', markersize=8),Line2D([0], [0], color='black',label='y+x+2.5=0')]
plt.legend(handles=legend_elements, title="Class")
plt.title("Polynomial kernel 3 * RBF Kernel sigma=0.1")
plt.show()

plt.scatter(Z[:, 0], Z[:, 1],c=colors)
plt.xlim(-2, -1.5)
plt.ylim(-1, -0.5)
x_line = np.linspace(-5, 15, 100)
y_line = -1 * x_line -5/2
plt.plot(x_line, y_line, 'k-', linewidth=2, label='y+x+2.5=0')
plt.xlabel("Projection value along 1st PC")
plt.ylabel("Projection value along 2nd PC")
legend_elements = [Line2D([0], [0], marker='o', color='w',label='-1', markerfacecolor='red', markersize=8),Line2D([0], [0], marker='o', color='w',label='1', markerfacecolor='blue', markersize=8),Line2D([0], [0], color='black',label='y+x+2.5=0')]
plt.legend(handles=legend_elements, title="Class")
plt.title("Polynomial kernel 3 * RBF Kernel sigma=0.1")
plt.show()

cm = np.zeros((2, 2))
D = Z[:, 0] + Z[:, 1] + 5/2
for i, d in enumerate(D):
    if d > 0:
        pred = -1
    else:
        pred = 1
    if pred == -1:
        if y[i] == -1:
            cm[0][0] += 1
        else:
            cm[0][1] += 1
    else:
        if y[i] == -1:
            cm[1][0] += 1
        else:
            cm[1][1] += 1

accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)

print(f"The accuracy of the decision line is {accuracy}")
print("Confusion matrix:")
print(cm)