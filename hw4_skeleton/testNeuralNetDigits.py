from numpy import loadtxt
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from nn import NeuralNet
from sklearn.grid_search import RandomizedSearchCV,GridSearchCV


filename1 = 'data/digitsX.dat'
data = loadtxt(filename1, delimiter=',')
X = data[:,:]
filename2 = 'data/digitsY.dat'
y = loadtxt(filename2, delimiter=',')

a = np.arange(100,1000,100)
b = np.arange(1.0,3.0,0.25)
c = np.logspace(-5,-3,5)

d= np.array([0.003])
dicta = {'numEpochs': a, 'learningRate':b, 'lambdaa':c, 'layers':[[25]]}


clf = NeuralNet(layers = [25], lambdaa=0.0001, numEpochs = 750, learningRate=2.0)
clf.fit(X,y)

predicted = clf.predict(X)


acu = accuracy_score(y , predicted )

print(acu)