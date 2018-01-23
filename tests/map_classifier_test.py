import unittest
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn import model_selection
import numpy as np
import random
import map_classifier

STANDARD_RANDOM_SEED = 101
NP_RANDOM_SEED       = 1010
# Set standard random seed
random.seed(STANDARD_RANDOM_SEED)
# Set numpy random seed
np.random.seed(NP_RANDOM_SEED)

class MapClassifierTest(unittest.TestCase):

  def test_train_accuracy(self):
    X, y = load_iris(return_X_y=True)
    clf = map_classifier.MAPClassifier()
    clf.fit(X, y)
    # Predict train set
    y_pred = clf.predict(X)
    # Train accuracy
    train_accuracy = metrics.accuracy_score(y, y_pred)
    # Train accuracy should > 0.90
    self.assertGreater(train_accuracy, 0.90)

  def test_k_fold_cv(self):
    X, y = load_iris(return_X_y=True)
    kf = model_selection.KFold(n_splits=5, shuffle=True)
    clf = map_classifier.MAPClassifier()

    test_accuracies = []
    for train_index, test_index in  kf.split(X):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)

      test_accuracy = metrics.accuracy_score(y_test, y_pred)
      test_accuracies.append(test_accuracy)

    avg_test_accuracy = np.mean(test_accuracies)
    # Average test accuracy should > 0.65
    self.assertGreater(avg_test_accuracy, 0.65)


def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(MapClassifierTest))
  return suite
