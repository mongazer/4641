'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score


class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        #TODO
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
        self.betta = []
        self.adaboost = []
    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        #TODO
        n,d = X.shape
        weight = [1.0/n]*n
        print(X.shape)
        print(y.shape)
        #print(weight)
        for i in range(self.numBoostingIters):
        #for i in range(10):
            temptree = tree.DecisionTreeClassifier(max_depth=self.maxTreeDepth)
            temptree.fit(X, y, sample_weight=weight)
            #print(y)
            predX = temptree.predict(X)
            #print(predX)
            predX-=y
            #print(y)
            predX[predX!=0] = 1
            #print(predX)
            temperr = (predX*weight).sum()
            #print(weight)
            #print(temperr)            
            tempbeta = 0.5*(np.log(np.divide(1-temperr, temperr)) + np.log(self.numBoostingIters-1))
            #print(tempbeta)
            newweight = weight*np.exp(tempbeta*predX)
            self.betta.append(tempbeta)
            self.adaboost.append(temptree)
            newweight/=(newweight.sum())
            weight = newweight
            #print(weight)





    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        #TODO
        n,d = X.shape
        rt = np.zeros(n)
        res = np.zeros((10, n))
        betta = np.matrix(self.betta)
        #print(betta)
        tb = np.zeros((self.numBoostingIters,n))
        for i in range(self.numBoostingIters):
            pred = self.adaboost[i].predict(X)
            tb[i] = pred
        #print(tb)
        for i in range(10):
            temp = np.zeros((self.numBoostingIters,n))
            temp[tb==i] = 1
            #print(temp)
            res[i] = betta*temp
        #print(res)
        #print(np.argmax(res, axis=0))
        return np.argmax(res, axis=0)
            #print(res)
        #return res.max(axis=1)

if __name__ == "__main__":
    # Load Data
    filename = 'data/challengeTrainLabeled.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 0:-1]
    print(X.shape)
    y = np.array([data[:, -1]])[0]
    print(y.shape)
    n,d = X.shape
    idx = np.arange(n)
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=2)
    modelBoostedDT.fit(X,y)
    accuracyBoostedDT = accuracy_score(y, modelBoostedDT.predict(X))
    print "Boosted Decision Tree Accuracy = "+str(accuracyBoostedDT)
    # filename2 = 'data/challengeTestUnlabeled.dat'
    # Xtest = np.loadtxt(filename2, delimiter=',')
    # #print(Xtest.shape)
    # np.set_printoptions(threshold=np.nan)
    # ypred = modelBoostedDT.predict(Xtest)
    # print(','.join([str(x) for x in ypred]))
