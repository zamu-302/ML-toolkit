import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import KernelCenterer
from sklearn.model_selection import cross_validate
from cvxopt import matrix, solvers

class SVM():
    def __init__(self,rbf,gamma,C=None):
        self.rbf=rbf
        if self.rbf:
            if not C:
                raise ValueError("Please input the Constant C value.")
            self.C=C
        self.gamma=gamma
        self.linear=None
        

    
    def _prepareData(self,Y):
        Y=np.reshape(Y,(-1,1))
        Y=np.where(Y>0,1,-1)
        return Y

    

    def fit(self,X,Y):
        Y=self._prepareData(Y)#making y 1 or -1
        self.K=self.kerenel(X)# the kerenel value in inf dimenesion

        self.alpha=self.Calc_alpha(Y,self.C)# finds the maximimum value of alpha with Constriants

        self.w=np.sum(self.alpha@Y@X)

        support_indices=[i for i in range(len(self.alpha)) if 1e-5<=self.alpha[i]<=self.C]
        
        for indices in support_indices:
            self.b+=Y[indices]-np.sum(self.alpha[i]*Y[i]*self.K[i,indices] for i in range(len(self.alpha)))
        self.b/=len(support_indices)

    


    def Calc_alpha(self,y,C):

        n = self.K.shape[0]
        P = matrix(np.outer(y, y) @ self.K)#yi*yj@K
        q = matrix(-np.ones(n))
        # Inequality constraints 0 <= alpha <= C
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = matrix(np.hstack((np.zeros(n), C*np.ones(n))))
        # Equality constraint sum(alpha_i * y_i) = 0
        A = matrix(y.reshape(1,-1))
        b = matrix(np.zeros(1))
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x']).flatten()
        return alpha



    

    def kerenel(self,X):
        n=X.shape[0]
        k=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                diff=(X[i]-X[j])
            k[i,j]=np.exp(-self.gamma@((diff)**2))
        return k


                 
    
    def predict(self,X):
        if self.linear:
            return np.sign((self.w.T@X)+self.b)
        else:
            return np.sign(np.sum(self.alpha@X@self.K)+self.b)


        
        


        