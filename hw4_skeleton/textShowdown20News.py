import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity




#preprocessing
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)	
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
count_vect = CountVectorizer(stop_words="english")
tfidf_transformer = TfidfTransformer(norm=u'l2',sublinear_tf= True)

X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_test_counts = count_vect.transform(twenty_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)


#--------------NB-------------------------
start = time.time()
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)	
end = time.time()
nb_trainingTime = end-start

#training
nb_predicted_train = clf.predict(X_train_tfidf)
train_accuracy = metrics.accuracy_score(twenty_train.target, nb_predicted_train)
train_precision, train_recall, fscore, support= precision_recall_fscore_support(twenty_train.target, nb_predicted_train, average='macro')

#testing
nb_predicted = clf.predict(X_test_tfidf)
test_accuracy = metrics.accuracy_score(twenty_test.target, nb_predicted)
test_precision, test_recall, fscore, support= precision_recall_fscore_support(twenty_test.target, nb_predicted, average='macro')	

#output
print('nb train accuracy: ' + str(train_accuracy))
print('nb test accuracy: ' + str(test_accuracy))
print('nb train precision: ' + str(train_precision))
print('nb test precision: ' + str(test_precision))
print('nb train recall: ' + str(train_recall))
print('nb test recall: ' + str(test_recall))



#--------------SVM-------------------------
start = time.time()
clf = SVC(kernel = cosine_similarity).fit(X_train_tfidf,twenty_train.target)
end = time.time()
svm_trainingTime = end-start

#training
svm_predicted_train = clf.predict(X_train_tfidf)
train_accuracy = metrics.accuracy_score(twenty_train.target, svm_predicted_train)
train_precision, train_recall, fscore, support= precision_recall_fscore_support(twenty_train.target, svm_predicted_train, average='macro')	

#testing data
svm_predicted = clf.predict(X_test_tfidf)
test_accuracy = metrics.accuracy_score(twenty_test.target, svm_predicted)
test_precision, test_recall, fscore, support= precision_recall_fscore_support(twenty_test.target, svm_predicted, average='macro')	

#output
print('svm train accuracy: ' + str(train_accuracy))
print('svm test accuracy: ' + str(test_accuracy))
print('svm train precision: ' + str(train_precision))
print('svm test precision: ' + str(test_precision))
print('svm train recall: ' + str(train_recall))
print('svm test recall: ' + str(test_recall))

print('nb trainging time: '+ str(nb_trainingTime))
print('svm trainging time: '+ str(svm_trainingTime))
