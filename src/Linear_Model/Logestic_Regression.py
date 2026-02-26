import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LogisticRegression():
    def __init__(self,iteration,learning_rate):
        self.iteration=iteration
        self.learning_rate=learning_rate
        self.cost=[]
    
    

    def _prepareData(self,X,Y):
        X = np.array(X, dtype=np.float64)  
        Y = np.array(Y, dtype=np.float64)
        m=X.shape[0]
        X=np.hstack([np.ones((m,1)),X])
        Y=np.reshape(Y,(-1,1))
        self.theta=np.zeros((X.shape[1],1))
        return X,Y,m
    def fit(self,X,Y):
        X,Y,m=self._prepareData(X,Y)

        for i in range(self.iteration):
            h_hat=1/(1+np.exp(-(X@self.theta)))
            y_hat=(h_hat>=0.5).astype(int)
            gradient=(1/m)*(X.T)@(h_hat-Y)
            if i%100==0:
               self.cost.append(-(1/m) * np.sum(Y * np.log(h_hat+1e-9) + (1 - Y) * np.log(1 - h_hat+1e-9))) 
            self.theta=self.theta-self.learning_rate*gradient 
    def Loss(self):
        return self.cost

    
    def predict(self,X):
        m=X.shape[0]
        X=np.hstack([np.ones((m,1)),X])
        h_hat=1/(1+np.exp(-(X@self.theta)))
        y_hat= (h_hat>=0.5).astype(int)
        return y_hat
