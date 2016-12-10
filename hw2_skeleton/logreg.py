'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters

    def sigmoid(self, theta, xi):
        return 1/(1+np.exp(-xi.dot(theta)))

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        
        return (-y.T.dot(np.log(self.sigmoid(theta, X))))-(1-y).T.dot(1-np.log(self.sigmoid(theta, X)))+regLambda/2*theta.T.dot(theta)

    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        n, d = X.shape
        new_theta = np.c_[theta]       
        temp = self.sigmoid(theta, X)
        temp = temp-y
        #####################################################################################
        new_theta = temp.T.dot(X).T + regLambda*theta
        ##########################################################################################
        new_theta[0,0] -= regLambda*theta[0,0]
        return new_theta



    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n, d = X.shape
        X = np.c_[np.ones([n, 1]), X];
        theta = np.matrix(np.ones((d+1, 1)))
        for i in range(self.maxNumIters):            
            new_theta = self.computeGradient(theta,X,y,self.regLambda)
            if (np.linalg.norm(new_theta-theta)<=self.epsilon):
                break
            else:
                theta -= self.alpha*new_theta
        self.theta = theta

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n, d = X.shape
        X = np.c_[np.ones([n, 1]), X];   
        y = np.array(self.sigmoid(self.theta, X)).T[0]  
        for i in range(len(y)):
            y[i] = 1 if y[i]>0.5 else -1
        return y


