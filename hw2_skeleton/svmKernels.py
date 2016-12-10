"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 3
_gaussSigma = 0.1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    return np.array(np.power(X1.dot(X2.T)+np.ones((X1.shape[0], X2.shape[0])), _polyDegree))



def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1, d = X1.shape
    n2, d = X2.shape
    a = np.repeat(X1, n2, axis=0)
    b = np.tile(X2,(n1,1))
    c = a-b
    c = (c*c).sum(axis=1).reshape((n1, n2))
    c /= (-2*np.power(_gaussSigma,2))
    return np.exp(c)



