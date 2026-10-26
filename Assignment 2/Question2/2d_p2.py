import numpy as np
import pandas as pd

X = pd.read_csv("Dataset2Train.csv", header=None)
X = X.to_numpy()

Y = X[:, -1]
X = X[:, :-1]

m = X.shape[0]
n = X.shape[1]

X=X.T

def ridge(X,Y,m,n,l):
    A = X@X.T+l*np.eye(n)
    if np.linalg.det(A)==0:
        print("Not invertible")
        return False,1
    else:
        w = np.linalg.inv(A)@X@Y
        print("The ridge regression estimate for w:")
        print()
        print(w)
        return True,w

def test_w(w,flag):
    if not flag:
        print("Regression failed")

    else:
        X = pd.read_csv("Dataset2Test.csv", header=None)
        X = X.to_numpy()
    
        Y = X[:, -1]
        X = X[:, :-1]
    
        m = X.shape[0]
        n = X.shape[1]
    
        X = X.T
    
        e = (X.T)@w-Y
        e = np.dot(e,e)
        print(f"The error on test set due to the ridge regression estimator w is: {e}")

flag,w = ridge(X,Y,m,n,1)
test_w(w,flag)