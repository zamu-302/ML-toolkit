import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X,Y=make_classification(
    n_features=4,
    n_samples=10000,
    random_state=42,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.5

)


class GDA():
    def __init__(self):
        self.theta0=None
        self.theta1=None
    

    def _prepare_Data(self,X,Y):
        m,n=X.shape
        sigma=np.zeros((n,n))
        pi=(np.sum(Y)+1)/(len(Y)+2)#laplace smooting
        mu0=X[Y==0].mean(axis=0)
        mu1=X[Y==1].mean(axis=0)


        return m,sigma,pi,mu0,mu1
    def fit(self,X,Y):
        m,sigma,pi,mu0,mu1=self._prepare_Data(X,Y)
    
        
        diff0=(X[Y==0]-mu0)
        diff1=(X[Y==1]-mu1)
        sigma=(diff0.T@diff0 + diff1.T@diff1)
        
        
        sigma/=m
        sigma+=1e-6*np.eye(sigma.shape[0])
        self.theta1=np.linalg.inv(sigma)@(mu1-mu0)
        
        self.theta0=(
        np.log(pi/(1-pi)) 
        -0.5*((mu1.T)@np.linalg.inv(sigma)@mu1)
         +0.5*((mu0.T)@np.linalg.inv(sigma)@mu0))# don't panic lol more on the docmentation
        

    def predict(self,X):
        h_hat= 1/(1+np.exp(-(X@self.theta1+self.theta0)))
        y_hat=(h_hat>=0.5).astype(int)
        return y_hat


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=GDA()
model.fit(x_train,y_train)
print("Accuracy: ",accuracy_score(y_test,model.predict(x_test)))

