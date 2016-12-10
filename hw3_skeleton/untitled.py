from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from boostedDT import BoostedDT
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score





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


modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=3)
modelBoostedDT.fit(X,y)
accuracyBoostedDT = accuracy_score(y, modelBoostedDT.predict(X))
print "Boosted Decision Tree Accuracy = "+str(accuracyBoostedDT)

modelBoostedDT2 = BoostedDT(numBoostingIters=100, maxTreeDepth=2)
modelBoostedDT2.fit(X,y)
accuracyBoostedDT2 = accuracy_score(y, modelBoostedDT2.predict(X))
print "Boosted Decision Tree Accuracy 2 = "+str(accuracyBoostedDT2)

ada = AdaBoostClassifier()
ada.fit(X,y)
accuracyada = accuracy_score(y, ada.predict(X))
print "Adaboost Accuracy = "+str(accuracyada)



rfc =  RandomForestClassifier(n_estimators=10)
rfc.fit(X,y)
rfcacu = accuracy_score(y, rfc.predict(X))
print "Random Forest Accuracy = "+str(rfcacu)


clf = SVC()
clf.fit(X, y) 
clfacu = accuracy_score(y, clf.predict(X))
print "SVM Accuracy = "+str(clfacu)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)
neighacu = accuracy_score(y, neigh.predict(X))
print "KNN Accuracy = "+str(neighacu)

filename2 = 'data/challengeTestUnlabeled.dat'
Xtest = np.loadtxt(filename2, delimiter=',')
#print(Xtest.shape)
np.set_printoptions(threshold=np.nan)
ypred = modelBoostedDT.predict(Xtest)
ypred2 = neigh.predict(Xtest)
ypred3 = modelBoostedDT2.predict(Xtest)
ypred4 = rfc.predict(Xtest)

# print(np.count_nonzero(ypred2-ypred))
# print(np.count_nonzero(ypred3-ypred))
# print(np.count_nonzero(ypred3-ypred2))
# print(np.count_nonzero(ypred4-ypred))
# print(np.count_nonzero(ypred4-ypred2))
print(','.join([str(int(x)) for x in ypred]))
