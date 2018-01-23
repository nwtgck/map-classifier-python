import unittest
from sklearn.datasets import load_iris
from sklearn import metrics

import map_classifier

class MapClassifierTest(unittest.TestCase):

  def test_train_accuracy(self):
    X, y = load_iris(return_X_y=True)
    clf = map_classifier.MAPClassifier()
    clf.fit(X, y)
    # Predict train set
    pred_y = clf.predict(X)
    # Train accuracy
    train_accuracy = metrics.accuracy_score(y, pred_y)
    # Train accuracy should > 0.95
    self.assertGreater(train_accuracy, 0.95)


def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(MapClassifierTest))
  return suite
