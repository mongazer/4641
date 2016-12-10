import numpy as np

def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    n = len(x1)

    poly = np.c_[x1, x2, np.matrix(np.zeros((n, 25)))]
    cnt = 2
    for i in range(2, 7):
        for j in range(i+1):
            for k in range(n):
                poly[k, cnt] = np.power(poly[k,1], j)*np.power(poly[k, 0], i-j)
            cnt +=1

    return poly