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
        self.Y=np.reshape(Y,(-1,1))
        self.Y=np.where(Y>0,1,-1)
        return self.Y

    

    def fit(self,X,Y):
        Y=self._prepareData(Y)#making y (1 or -1)
        if self.rbf:
            self.K=self.kerenel(X)# the kerenel value in inf dimenesion
        else:
            self.K=X@X.T

        self.alpha=self.Calc_alpha(Y,self.C)# finds the maximimum value of alpha with Constriants
        if not self.rbf:
            self.w=np.sum(self.alpha.reshape(-1,1)*Y*X,axis=0)

        support_indices=[i for i in range(len(self.alpha)) if 1e-5<=self.alpha[i]<=self.C]
        self.b=0
        
        for indices in support_indices:
            term=np.sum(self.alpha*Y.flatten()*self.K[:,indices])
            self.b+=Y[indices]-term
        self.b/=len(support_indices)

        self.X=X

    


    def Calc_alpha(self,y,C):

        n = self.K.shape[0]
        P = matrix((np.outer(y, y) * self.K),tc='d')#yi*yj*K
        q = matrix(-np.ones(n),tc='d')
        # Inequality constraints 0 <= alpha <= C
        G = matrix(np.vstack((-np.eye(n), np.eye(n))),tc='d')
        h = matrix(np.hstack((np.zeros(n), C*np.ones(n))),tc='d')
        # Equality constraint sum(alpha_i * y_i) = 0
        A = matrix(y.reshape(1,-1),tc='d')
        b = matrix(np.zeros(1),tc='d')
       
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x']).flatten()
        return alpha
    



    

    def kerenel(self,X):
        n=X.shape[0]
        k=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                diff=(X[i]-X[j])
                k[i,j]=np.exp(-self.gamma*(diff@diff))
        return k


                 
    
    def predict(self,X):
        if not self.rbf:
            return np.sign((X@self.w)+self.b)
        else:
            y_hat=[]
            for test in X:
                k_step = np.exp(-self.gamma * np.sum((self.X - test)**2, axis=1))#radical Kernel Trick for inf dim
                prediction = np.sum(self.alpha * self.Y.flatten() * k_step) + self.b
                y_hat.append(prediction)



            return np.sign(np.array(y_hat))


        
        


        