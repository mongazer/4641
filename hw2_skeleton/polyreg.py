'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np



#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class LinearRegressionClosedForm:

    def __init__(self, regLambda = 1E-8):
        '''
        Constructor
        '''
        self.regLambda = regLambda;        

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-d array
                y is an n-by-1 array
            Returns:
                No return value
        '''
        n = len(X)
        
        # add 1s column
        Xex = np.c_[np.ones([n, 1]), X];
        #print(Xex)
        
        n,d = Xex.shape
        d = d-1  # remove 1 for the extra column of ones we added to get the original num features
        
        # construct reg matrix
        regMatrix = self.regLambda * np.eye(d + 1)
        regMatrix[0,0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.pinv(Xex.T.dot(Xex) + regMatrix).dot(Xex.T).dot(y);
        #print(self.theta)
        
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        n = len(X)
        
        # add 1s column
        Xex = np.c_[np.ones([n, 1]), X];

        # predict
        return Xex.dot(self.theta.T);


class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        self.lg = LinearRegressionClosedForm(regLambda=regLambda)
        self.degree = degree

    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        #print(X.shape)
        n = len(X)
        newX = np.matrix(np.ones([n, degree]))
        
        for i in range(n):
            newX[i, 0] = X[i]
            for j in range(1, degree):
                newX[i, j] = newX[i, j-1]*X[i]        
        return newX
    
    def standize1(self, X):
        n, d = X.shape
        self.model = np.zeros((d, 2))
        for i in range(d):               
            #std = np.max(X[:,i])-np.min(X[:,i])
            std = np.std(X[:,i])
            self.model[i,0] = std
            if (std==0):
                continue
            m = np.mean(X[:,i])
            self.model[i,1] = m
            for j in range(n):
                X[j, i] = (X[j, i]-m)/std
        return X

    def standize2(self, X):
        n, d = X.shape
        for i in range(d):
            std = self.model[i,0]
            if (std==0):
                continue
            m = self.model[i,1]
            for j in range(n):
                X[j, i] = (X[j, i]-m)/std
        return X

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        #standize
        ployX = self.polyfeatures(X, self.degree)
        stanX = self.standize1(ployX)
        self.lg.fit(stanX, y)


        
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        # TODO
        ployX = self.polyfeatures(X, self.degree)
        stanX = self.standize2(ployX)

        return self.lg.predict(stanX)



#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------

def error(est, y):
    n = len(y)
    rt = 0
    for i in range(n):
        a = est[i]-y[i]
        rt += a*a
    return np.divide(rt, n)



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''

    n = len(Xtrain);
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))    
    pr = PolynomialRegression(regLambda=regLambda, degree=degree)
    
    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape
    for i in range(n):
        pr.fit(Xtrain[0:(i+1)], Ytrain[0:(i+1)])
        rt1 = pr.predict(Xtrain[0:(i+1)])
        rt2 = pr.predict(Xtest)
        errorTrain[i] = error(rt1, Ytrain[0:(i+1)])
        errorTest[i] = error(rt2, Ytest)

    
    return (errorTrain, errorTest)
