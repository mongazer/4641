"""
======================
Script to Explore SVMs
======================

Simple script to explore SVM training with varying C

Example adapted from scikit_learn documentation by Eric Eaton, 2014

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.grid_search import RandomizedSearchCV,GridSearchCV


# load the data
filename = 'data/svmTuningData.dat'
allData = np.loadtxt(filename, delimiter=',')

X = allData[:,:-1]
Y = allData[:,-1]
cc=np.array([401])
#cc = np.arange(1,1000)
a = np.matrix(np.logspace(-4,4,9))
b = np.matrix(np.array([1,3,6]))
#d = np.array(b.T.dot(a).T).flatten()
d= np.array([0.003])
dicta = {'C': cc, 'gamma':d}



# train the SVM
print "Training the SVM"
clf = svm.SVC()

learner = GridSearchCV(clf, dicta, cv=10)

learner.fit(X, Y)
print(learner)
print(learner.best_score_)
print(learner.best_estimator_)

C = learner.best_estimator_.C
var = learner.best_estimator_.gamma

print ""
print "Testing the SVM"

h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = learner.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.title('SVM decision surface with C = '+str(C) + ' var = '+str(var))
plt.axis('tight')
plt.show()
