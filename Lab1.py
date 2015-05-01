import pandas as pd
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('https://s3-us-west-2.amazonaws.com/ga-dat-2015-suneel/datasets/bank-additional-full.csv', delimiter=';')

#show a preview of the data
df.head()

# split the data into training and test set randomly
# 30% of the data is used for training and the other 70% is used for testing
test_idx = np.random.uniform(0, 1, len(df)) <= 0.3
train = df[test_idx==True]
test = df[test_idx==False]

#run knn classifier using age and consumer price index

features = ['age', 'cons.price.idx']
results = []
# range(1, 51, 2) = [1, 3, 5, 7, ...., 49]
for n in range(1, 51, 2):
    clf = KNeighborsClassifier(n_neighbors=n)
    # train the classifier
    clf.fit(train[features], train['y'])
    # then make the predictions
    preds = clf.predict(test[features])
    # very simple and terse line of code that will check the accuracy
    # documentation on what np.where does: http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
    # Here is a simple example: suppose our predictions where [True, False, True] and the correct values were [True, True, True]
    # The next line says, create an array where when the prediction = correct value, the value is 1, and if not the value is 0.
    # So the np.where would, in this example, produce [1, 0, 1] which would be summed to be 2 and then divided by 3.0 to get 66% accuracy
    accuracy = np.where(preds==test['y'], 1, 0).sum() / float(len(test))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)

    results.append([n, accuracy])
    
#again, using different features

features = ['age', 'euribor3m', 'campaign', 'cons.price.idx', ]
results = []
for n in range(1, 51, 2):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(train[features], train['y'])
    preds = clf.predict(test[features])
    accuracy = np.where(preds==test['y'], 1, 0).sum() / float(len(test))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)

    results.append([n, accuracy])

#knn classifier using all numerical columns
features = ['age', 'campaign', 'duration', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
results = []
for n in range(1, 51, 2):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(train[features], train['y'])
    preds = clf.predict(test[features])
    accuracy = np.where(preds==test['y'], 1, 0).sum() / float(len(test))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)
    results.append([n, accuracy])

#show a graph of accuracy and K
results = pd.DataFrame(results, columns=["n", "accuracy"])
pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()

#Logistic Regression
from sklearn.linear_model import LogisticRegression
features = ['age', 'campaign', 'duration', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
clf = LogisticRegression()
clf.fit(train[features], train['y'])
preds = clf.predict(test[features])
accuracy = np.where(preds==test['y'], 1, 0).sum() / float(len(test))
print 'logistic regression accuracy'
print accuracy

#It seems that KNN is (very) slightly more accurate

#Logistic Regression on subsets
from sklearn.linear_model import LogisticRegression
features = ['age', 'euribor3m', 'campaign', 'cons.price.idx', ]
clf = LogisticRegression()
clf.fit(train[features], train['y'])
preds = clf.predict(test[features])
accuracy = np.where(preds==test['y'], 1, 0).sum() / float(len(test))
print 'logistic regression accuracy'
print accuracy

#Logistic Regression on subsets
from sklearn.linear_model import LogisticRegression
features = ['age', 'euribor3m', 'duration', 'campaign', 'cons.price.idx', 'previous']
clf = LogisticRegression()
clf.fit(train[features], train['y'])
preds = clf.predict(test[features])
accuracy = np.where(preds==test['y'], 1, 0).sum() / float(len(test))
print 'logistic regression accuracy'
print accuracy
