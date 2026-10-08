import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Dataset2.xlsx")
X = df.to_numpy()

X=X.T
n=X.shape[0]
m=X.shape[1]

np.random.seed(19)

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

def get_voronoi_lines(C):

    L = []

    xmin = -10
    xmax = 10
    ymin = -10
    ymax = 10

    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            mx = (C[i][0] + C[j][0]) / 2
            my = (C[i][1] + C[j][1]) / 2
            dx = -(C[j][1] - C[i][1])
            dy = C[j][0] - C[i][0]

            tmin = -float('inf')
            tmax = float('inf')

            valid = True

            for k in range(len(C)):
                if k == i or k == j:
                    continue

                ax = C[k][0] - C[i][0]
                ay = C[k][1] - C[i][1]
                b = (C[k][0]**2 + C[k][1]**2- C[i][0]**2 - C[i][1]**2- 2 * (mx * ax + my * ay))
                a = 2 * (dx * ax + dy * ay)

                if a == 0:
                    if b < 0:
                        valid = False
                        break
                elif a > 0:
                    t = b / a
                    if t < tmax:
                        tmax = t
                else:
                    t = b / a
                    if t > tmin:
                        tmin = t

            if not valid:
                continue
            if dx != 0:
                t1 = (xmin - mx) / dx
                t2 = (xmax - mx) / dx
                if t1 > t2:
                    temp = t1
                    t1 = t2
                    t2 = temp
                if t1 > tmin:
                    tmin = t1
                if t2 < tmax:
                    tmax = t2
            else:
                if mx < xmin or mx > xmax:
                    valid = False
            if dy != 0:
                t1 = (ymin - my) / dy
                t2 = (ymax - my) / dy
                if t1 > t2:
                    temp = t1
                    t1 = t2
                    t2 = temp
                if t1 > tmin:
                    tmin = t1
                if t2 < tmax:
                    tmax = t2
            else:
                if my < ymin or my > ymax:
                    valid = False
            if valid and tmin <= tmax:
                x1 = mx + tmin * dx
                y1 = my + tmin * dy
                x2 = mx + tmax * dx
                y2 = my + tmax * dy

                L.append([[x1, y1],[x2, y2]])

    return L

def plot_clust_voronoi(Z,X,C,L):

    c = ['red', 'blue', 'green', 'orange', 'gold']
    colors = [c[v] for v in Z]
    plt.scatter(X[0, :],X[1, :],color=colors)
    
    for i, v in enumerate(C):
        plt.scatter(v[0],v[1],marker='X',s=200,color=c[i],edgecolors='black',linewidths=1.5,label=f'Center {i+1}')
        
    for li in L:
        plt.plot([li[0][0], li[1][0]],[li[0][1], li[1][1]],color='black')

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()
    plt.grid(True)
    plt.show()

K=[2,3,4,5]

for k in K:
    Z,E,C = Lloyds(X,n,m,k)
    L = get_voronoi_lines(C)
    plot_clust_voronoi(Z,X,C,L)