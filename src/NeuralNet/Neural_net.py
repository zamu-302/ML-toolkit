import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))
def softmax(z):
    return np.exp(z)/(1+np.exp(z))
def Rlu(z):
    return max(0,z)

def int_to_onehot(y,num_labels):
    ary=np.zeros((y.shape[0],num_labels))
    for i,val in enumerate(y):
        ary[i,val]=1
    return ary

class NeuralNetMLP:
    def __init__(self,num_features,num_hidden_layer,num_classes,activation_func=sigmoid,random_seed=42):
        super().__init__()
        self.num_classes=num_classes
        self.activation_func=activation_func

       #hidden
        rng=np.random.RandomState(random_seed) 
        self.weight_h=rng.normal(loc=0.0,scale=0.1,size=(num_hidden_layer,num_features))
        self.bias_h=np.zeros(num_hidden_layer)


        #ouput
        self.weight_out=rng.normal(loc=0.0,scale=0.1,size=(num_classes,num_hidden_layer))
        self.bias_out=np.zeros(num_classes)


    def forward(self,x):
            #input layer: x dot W.T==[n_examples,n_features] dot [n_hidden,n_features].T 
            #ouput layer: [n_examples,n_hidden]

            z_h=np.dot(x,self.weight_h.T)+self.bias_h
            a_h=self.activation_func(z_h)



            #output layer

            z_out=np.dot(a_h,self.weight_out.T)+self.bias_out
            a_out=self.activation_func(z_out)
            return a_h,a_out
        
    def backward(self,x,a_h,a_out,y):

            y_onehot=int_to_onehot(y,self.num_classes)

            # dloss/dout weight
            d_loss_d_a_out=2*(a_out-y_onehot)/y.shape[0]
            d_a_out__d_z_out=a_out*(1-a_out)

            delta_out=d_loss_d_a_out*d_a_out__d_z_out

            d_z_out__dw_out=a_h
            d_loss__dw_out=np.dot(delta_out.T,d_z_out__dw_out)
            d_loss__db_out=np.sum(delta_out,axis=0)

            #dloss/hidden dout weight

            d_z_out__a_h = self.weight_out
        
            # output dim: [n_examples, n_hidden]
            d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
            
            # [n_examples, n_hidden]
            d_a_h__d_z_h = a_h * (1. - a_h) # sigmoid derivative
            
            # [n_examples, n_features]
            d_z_h__d_w_h = x
            
            # output dim: [n_hidden, n_features]
            d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
            d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)


            return (d_loss__dw_out, d_loss__db_out, 
                d_loss__d_w_h, d_loss__d_b_h)
    






        


