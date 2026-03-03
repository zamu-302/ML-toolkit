import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import KernelCenterer
from sklearn.model_selection import cross_validate

class SVM():
    def __init__(self,kernel,C):
        self.kernel=kernel
        self.cv=cross_validate(cv=10)
        self.C=C
    
    def _prepareData(self,X,Y):
        Y=np.reshape(Y,(-1,1))
        pass

    def fit(self,X,Y):
        pass

        