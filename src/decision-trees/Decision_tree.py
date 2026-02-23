import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Node():
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,info_gain=None,value=None):
        self.feature_index=feature_index
        self.threshold=threshold
        self.left=left
        self.right=right
        self.info_gain=info_gain
        self.value=value#for leaf node
class Desision_Tree():
    def __init__(self,min_sample_split=20,max_depth=2):
        self.root=None  # The root of Tree

        self.min_sample_split=min_sample_split
        self.max_depth=max_depth
    

    def build_tree(self,data_set,curr_depth=0):
        X,Y=data_set[:,:-1],data_set[:,-1]
        sample_num,feature_num=np.shape(X)# return the num of col and row (features,number of sample)

        #Base Case (if max depth is reached or if sample less than min split(Avoiding overfitting))

        if sample_num>=self.min_sample_split and curr_depth<self.max_depth:
            best_split=self.get_split(data_set,sample_num,feature_num)  #looks for the best split and return a dictionary

            if best_split and best_split['info_gain']>0:#cause we don't wanna split more a pure dataset
                left_subtree=self.build_tree(best_split["left_dataset"],curr_depth+1)
                right_subtree=self.build_tree(best_split["right_dataset"],curr_depth+1)
                return Node(best_split["feature_index"],best_split["threshold"],left_subtree,right_subtree,best_split["info_gain"])
        leaf_value=self.calc_leaf_value(Y)
        return Node(value=leaf_value)
    
    def get_split(self,data_set,sample_num,feature_num):
        bestSplit={}
        max_info_gain=-float("inf")


        for feature_index in range(feature_num):
            feature_value=data_set[:,feature_index]
            possible_threshold=np.unique(feature_value)
            for threshold in possible_threshold:
                dataset_left,dataset_right=self.split(data_set,feature_index,threshold)

                if len(dataset_left)>0 and len(dataset_right)>0:
                    y,left_y,right_y=data_set[:,-1],dataset_left[:,-1],dataset_right[:,-1]

                    cur_info=self.information_gain(y,left_y,right_y,"gini")

                    if cur_info>max_info_gain:#updating the data for a new max
                        bestSplit["feature_index"]=feature_index
                        bestSplit["threshold"]=threshold
                        bestSplit["left_dataset"]=dataset_left
                        bestSplit["right_dataset"]=dataset_right
                        bestSplit["info_gain"]=cur_info
                        max_info_gain=cur_info
        return bestSplit
    
    def split(self,data_set,feature_index,threshold):
        left_dataset=np.array([row for row in data_set if row[feature_index]<=threshold])
        right_dataset=np.array([row for row in data_set if row[feature_index]> threshold])
        return left_dataset,right_dataset



    def information_gain(self,parent,left_child,right_child,mode="entropy"):
        weight_l=len(left_child)/len(parent)
        weight_r=len(right_child)/len(parent)
        if mode=="gini":
            gain=self.gini_index(parent)-(weight_l*self.gini_index(left_child)+weight_r*self.gini_index(right_child))# calculating the weighted gini
        else:
            gain=self.entropy(parent)-(weight_l*self.entropy(left_child)+weight_r*self.entropy(right_child))
        return gain



    def entropy(self,y):
        class_label=np.unique(y)
        entrop=0
        for cls in class_label:
            p_cls=len(y[y==cls])/len(y)
            entrop+= -p_cls*np.log2(p_cls) #entropy= summation of (-p(x)*log(px))
        return entrop
    def gini_index(self,y):
        class_label=np.unique(y)    #gives us the unique elements 
        gini=0
        for cls in class_label:
            p_cls=len(y[y==cls])/len(y) #probablity of each class
            gini+=p_cls**2
        return 1-gini
    def calc_leaf_value(self,y):
        y=list(y)
        return max(y,key=y.count)
    
    def fit(self,X,Y):
        data_set=np.concatenate((X,Y),axis=1)
        self.root=self.build_tree(data_set)

    def predict(self,X):
        prediction=[self.make_prediction(x,self.root) for x in X]
        return prediction
    
    def make_prediction(self,x,tree):
        if tree.value!=None :
            return tree.value
        feature_val=x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x,tree.left)
        else:
            return self.make_prediction(x,tree.right)

data=pd.read_csv("iris.csv")
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=Desision_Tree(min_sample_split=3,max_depth=3)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(accuracy_score(y_train,model.predict(x_train)))







