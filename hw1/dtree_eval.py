'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score



def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape
    #configuration
    cons = [None , 1, 3]
    rt = []
    n,d = X.shape
    trials=100
    fold=10
    perlen = n/fold
    
    # shuffle the data
    for con in cons :
        res = np.zeros(trials*fold,dtype=np.float64)
        for i in range(trials):
            idx = np.arange(n)
            np.random.seed(13)
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]
            for j in range(fold):
                # split the data
                Xtrain = np.concatenate((X[0:j*perlen,:],X[(j+1)*perlen:n,:]))  # train on remaining instances
                Xtest = X[j*perlen:(j+1)*perlen,:]
                ytrain = np.concatenate((y[0:j*perlen],y[(j+1)*perlen:n]))  # test on j-th fold
                ytest = y[j*perlen:(j+1)*perlen]

                # train the decision tree
                clf = tree.DecisionTreeClassifier(max_depth=con)
                clf = clf.fit(Xtrain,ytrain)

                # output predictions on the remaining data
                y_pred = clf.predict(Xtest)

                # compute the training accuracy of the model
                acu = accuracy_score(ytest, y_pred)

                # save current accuracy for further computing
                res[j+i*fold] = acu

        rt.append(np.mean(res))
        rt.append(np.std(res))


    

    meanDecisionTreeAccuracy = rt[0]
    
    
    # TODO: update these statistics based on the results of your experiment
    stddevDecisionTreeAccuracy = rt[1]
    meanDecisionStumpAccuracy = rt[2]
    stddevDecisionStumpAccuracy = rt[3]
    meanDT3Accuracy = rt[4]
    stddevDT3Accuracy = rt[5]

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats


    



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print "Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")"
    print "Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")"
    print "3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")"
# ...to HERE.
