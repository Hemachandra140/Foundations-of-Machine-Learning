import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

X = pd.read_csv("Dataset1.csv", header=None)
X = X.to_numpy()

m = X.shape[0]
n = X.shape[1]

np.random.seed(19)

p = np.zeros(n)
for i in range(m):
    p+=X[i, :]
p/=m
print("Assume each entry is identically distributed bernoulli random variable")
print("Sample Proportion for each entry")
print(p)
print("Average of distribution of sample proportion")
u=sum(p)/n
print(u)
print("Variance of distribution of sample proportion")
s=0
for i in range(n):
    s+=(p[i]-u)**2
s/=n-1
print(s)

bins = np.arange(0, 1.1, 0.1)
plt.figure(figsize=(10, 6))

plt.hist(p, bins=bins, density=True, alpha=0.6,edgecolor='black', label='Sample proportions')
 
x = np.linspace(0, 1, 500)
normal = (1 / np.sqrt(2 * np.pi * s)) * np.exp(-((x - u) ** 2) / (2 * s))

plt.plot(x, normal, linewidth=2,label='Expected Normal distribution')
plt.xlabel('Sample proportion')
plt.ylabel('Density')
plt.title('Distribution of Sample Proportions vs Normal Distribution')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


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
    it_err=[]

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
            return T,P,it_err
        
        P=P1.copy()
        T=T1.copy()

        logl=0
        for i in range(m):
            t2=0
            for j in range(K):
                t1=1
                for k in range(n):
                    t1*=(P[j][k])**(X[i][k])*(1-P[j][k])**(1-X[i][k])
                t2+=T[j]*t1
            logl+=np.log(t2)
        it_err.append(logl)


all_logl = []

for i in tqdm(range(100), desc="EM runs"):
    T, P, it_err = EM(X, m, n, 4)
    all_logl.append(it_err)

max_len = max(len(x) for x in all_logl)

for x in all_logl:
    if len(x) < max_len:
        x.extend([x[-1]] * (max_len - len(x)))

avg_logl = np.mean(all_logl, axis=0)

plt.plot(range(1, max_len + 1), avg_logl)
plt.xlabel("Iteration")
plt.ylabel("Average Log-Likelihood")
plt.title("Average Log-Likelihood over 100 Initializations")
plt.show()

plt.plot(range(1, 21), avg_logl[:20])
plt.xlabel("Iteration")
plt.ylabel("Average Log-Likelihood")
plt.title("Average Log-Likelihood over 100 Initializations")
plt.xlim(0, 20)
plt.show()