'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from sklearn import preprocessing

class NeuralNet:

    def __init__(self, layers,learningRate=2.0,  epsilon=0.12, numEpochs=100, lambdaa=0.0001):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.lambdaa = lambdaa


    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def forward(self,X):
        theta = self.theta
        res = X.copy()
        for i in range(1, self.layernum):            
            res = np.c_[np.ones(len(res)),res]
            self.a[i] = res
            res=self.sigmoid(res.dot(theta[i]))
        self.a[self.layernum] = res
        return res

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''

        #init 
        n,d = X.shape

        
        lb = preprocessing.LabelBinarizer()
        lb.fit(y)
        print(len(y))
        transy = lb.transform(y)
        self.lb = lb

        ynum = len(lb.classes_)
        print(ynum)

        layercnt = np.append(np.append(d, self.layers),ynum)
        self.layernum = len(layercnt)

        print(layercnt)

        theta = {}
        for i in range(1,len(layercnt)):
            theta[i] = np.random.random_sample(((layercnt[i-1] +1),layercnt[i])) * (self.epsilon*2) - self.epsilon
        self.theta = theta
        ############################################

        
        self.a = {}
        
        error={}

        gradient = {}

        for i in range(self.numEpochs):



            fy = self.forward(X)
            error[self.layernum] = fy-transy


            for j in range(self.layernum-1,1,-1):
                gg = self.a[j][:,1:]*(1-(self.a[j][:,1:]))
                error[j] = ((error[j+1].dot((self.theta[j][1:]).T))*gg)



            for j in range(1,self.layernum):
                if i==0:
                    gradient[j] = self.a[j].T.dot(error[j+1])
                else:
                    gradient[j] += self.a[j].T.dot(error[j+1])

                reg = self.theta[j]* self.lambdaa
                reg[0] = 0
                gradient[j] = gradient[j]/ n + reg
                self.theta[j] = self.theta[j] - self.learningRate * gradient[j]
       



    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        predicted = self.forward(X)
        return self.lb.inverse_transform(predicted)

        
    
    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        