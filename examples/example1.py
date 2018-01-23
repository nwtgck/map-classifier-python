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


# Load Iris data set
X, y = load_iris(return_X_y=True)

# Create a classifier
clf = map_classifier.MAPClassifier()

# K-fold CV
kf = model_selection.KFold(n_splits=5, shuffle=True)
test_accuracies = []
for train_index, test_index in  kf.split(X):
  # Get training and test sets
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  # Learn then model
  clf.fit(X_train, y_train)

  # Predict
  y_pred = clf.predict(X_test)

  # Calc accuracy
  test_accuracy = metrics.accuracy_score(y_test, y_pred)
  test_accuracies.append(test_accuracy)

# Calc average accuracy
avg_test_accuracy = np.mean(test_accuracies)

# Print the average accuracy
print(avg_test_accuracy)
