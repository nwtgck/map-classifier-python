# Maximum A Posteriori Classifier

[![Build Status](https://travis-ci.org/nwtgck/map-classifier-python.svg?branch=develop)](https://travis-ci.org/nwtgck/map-classifier-python) [![Coverage Status](https://coveralls.io/repos/github/nwtgck/map-classifier-python/badge.svg?branch=develop)](https://coveralls.io/github/nwtgck/map-classifier-python?branch=develop) 

A classifier use Maximum A Posteriori (MAP) which is compatible with [scikit-learn](http://scikit-learn.org/).


## Installation

```bash
pip3 install --upgrade git+https://github.com/nwtgck/map-classifier-python.git
```

## Example Usage

```python
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn import model_selection
import numpy as np
import map_classifier


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
```

You can find examples in [examples](examples).