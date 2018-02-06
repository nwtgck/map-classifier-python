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

def eq_dict(d1, d2):
  """
  Compare whether d1 and d2 are the same or not
  :param d1:
  :param d2:
  :return:
  """
  for k in d2.keys(): # NOTE: d2
    # Just assign the same value (This is for defaultdict)
    d1[k] = d1[k]
  for k in d1.keys(): # NOTE: d1
    # Just assign the same value (This is for defaultdict)
    d2[k] = d2[k]

  return set(d1.items()) == set(d2.items()) # (from: https://stackoverflow.com/a/4527978/2885946)

class MapClassifierTest(unittest.TestCase):

  def test_train_accuracy(self):
    X, y = load_iris(return_X_y=True)
    clf = map_classifier.MAPClassifier()
    clf.fit(X, y)
    # Predict train set
    y_pred = clf.predict(X)
    # Train accuracy
    train_accuracy = metrics.accuracy_score(y, y_pred)
    print("train_accuracy:", train_accuracy)
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
    print("avg_test_accuracy:", avg_test_accuracy)
    # Average test accuracy should > 0.65
    self.assertGreater(avg_test_accuracy, 0.65)


  def test_fit(self):

    # NOTE: I splitted 0th-dimension and 1th-dimension for readability

    # 0th-dimension of feature
    X_j0 = [
      1, 1, 2, 3,   2, 3, 2, 2,   5, 4, 5, 3
    ]
    # 1th-dimension of feature
    X_j1 = [
      2, 2, 3, 2,   4, 4, 4, 3,   4, 4, 3, 3
    ]

    # Class labels
    y = [
       0, 0, 0, 0,   1, 1, 1, 1,   2, 2, 2, 2
    ]

    # Combine the 2 dimensions
    X = np.array(list(zip(X_j0, X_j1)))

    print(X)


    # Create a MAP classifier
    map_clf = map_classifier.MAPClassifier()
    # Learn
    map_clf.fit(X, y)

    # ==== START: Test for P_j(x_j) ====
    # P_0(x_0)
    # (0-th dimension)
    j = 0
    expect = {
      1: 2/12,
      2: 4/12,
      3: 3/12,
      4: 1/12,
      5: 2/12
    }
    self.assertTrue(eq_dict(map_clf.p_j_x_dict[j], expect))

    # P_1(x_1)
    # (1-th dimension)
    j = 1
    expect = {
      1: 0/12,
      2: 3/12,
      3: 4/12,
      4: 5/12,
      5: 0/12
    }
    self.assertTrue(eq_dict(map_clf.p_j_x_dict[j], expect))
    # ==== END: Test for P_j(x_j) ====



    # ==== Start: Test for P_j(x_j | C_i) ====


    # ---- 0-th dimension ----

    # P_0(x_0 | C_0)
    # (0-th dimension, Class0)
    j  = 0
    Ci = 0
    expect = {
      1: 2/4,
      2: 1/4,
      3: 1/4,
      4: 0/4,
      5: 0/4
    }
    self.assertTrue(eq_dict(map_clf.p_j_Ci_x_dict[j][Ci], expect))

    # P_0(x_0 | C_1)
    # (0-th dimension, Class1)
    j  = 0
    Ci = 1
    expect = {
      1: 0 / 4,
      2: 3 / 4,
      3: 1 / 4,
      4: 0 / 4,
      5: 0 / 4
    }
    self.assertTrue(eq_dict(map_clf.p_j_Ci_x_dict[j][Ci], expect))

    # P_0(x_0 | C_2)
    # (0-th dimension, Class2)
    j  = 0
    Ci = 2
    expect = {
      1: 0 / 4,
      2: 0 / 4,
      3: 1 / 4,
      4: 1 / 4,
      5: 2 / 4
    }
    self.assertTrue(eq_dict(map_clf.p_j_Ci_x_dict[j][Ci], expect))

    # ---- 1-th dimension ----

    # P_1(x_1 | C_0)
    # (1-th dimension, Class0)
    j  = 1
    Ci = 0
    expect = {
      1: 0 / 4,
      2: 3 / 4,
      3: 1 / 4,
      4: 0 / 4,
      5: 0 / 4
    }
    self.assertTrue(eq_dict(map_clf.p_j_Ci_x_dict[j][Ci], expect))

    # P_1(x_1 | C_1)
    # (1-th dimension, Class1)
    j  = 1
    Ci = 1
    expect = {
      1: 0 / 4,
      2: 0 / 4,
      3: 1 / 4,
      4: 3 / 4,
      5: 0 / 4
    }
    self.assertTrue(eq_dict(map_clf.p_j_Ci_x_dict[j][Ci], expect))

    # P_1(x_1 | C_2)
    # (1-th dimension, Class2)
    j  = 1
    Ci = 2
    expect = {
      1: 0 / 4,
      2: 0 / 4,
      3: 2 / 4,
      4: 2 / 4,
      5: 0 / 4
    }
    self.assertTrue(eq_dict(map_clf.p_j_Ci_x_dict[j][Ci], expect))


def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(MapClassifierTest))
  return suite
