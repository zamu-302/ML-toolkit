import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class LinearRegression():
    def __init__(self,max_iteration=1000,learning_rate=0.01,theta=None):
        self.max_iteration=max_iteration
        self.learning_rate=learning_rate
        self.theta=theta
        self.loss_history=[]
     

    def _prepare_data(self,X,Y):
        self.mean=X.mean(axis=0)
        self.std=X.std(axis=0)
        X=(X-self.mean)/self.std
        Y=Y.reshape(-1,1)

        m=X.shape[0]
        X=np.hstack([np.ones((m,1)),X])
        self.theta=np.zeros((X.shape[1],1))
        return X,Y,m
    
    def fit(self,X,Y):
        X,Y,m=self._prepare_data(X,Y)
        iteration=0
        min_diff=1e-6

        while iteration<self.max_iteration:
            y_hat=X.dot(self.theta)
            gradient=(1/m)*(X.T)@(y_hat-Y)
            loss=((y_hat-Y)**2).mean()
            self.loss_history.append(loss)
            self.theta=self.theta-(self.learning_rate*gradient)
            if iteration>0 and abs(self.loss_history[-2]-loss)<min_diff:
                break
            iteration+=1

    def predict(self,X):
        m=X.shape[0]
        X=(X-self.mean)/self.std
        X=np.hstack([np.ones((m,1)),X])
        return X@self.theta





    
           

