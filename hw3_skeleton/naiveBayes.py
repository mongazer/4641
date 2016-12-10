'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from sklearn.preprocessing import normalize

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,d = X.shape
        yrange = np.unique(y)
        self.ycat = len(yrange)
        self.px = np.zeros((self.ycat, d))
        self.py = np.zeros(self.ycat)
        for i in yrange:
            sely = y[y==i]
            py = float(len(sely))/len(y)
            self.py[i] = py
            selx = X[y==i]
            if (self.useLaplaceSmoothing):
                px = (selx.sum(axis=0)+1)/(np.sum(selx)+d)
            else:
                px = selx.sum(axis=0)/np.sum(selx)
            self.px[i] = px
        self.px = np.log(self.px).T
        self.py = np.log(self.py)


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        res = X.dot(self.px) + self.py
        return np.argmax(res, axis = 1)


    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        res = X.dot(self.px) + self.py
        return normalize(res, axis=1, norm='l1')




        
        
        