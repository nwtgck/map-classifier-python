# Bayesian Maximum a posteriori estimation (MAP)

from collections import defaultdict
import numpy as np
import sklearn

class MAPClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        assert (len(X) == len(y))

        self.Ci_list       = sorted(list(set(y)))

        count_j_dict    = defaultdict(lambda: 0)
        count_j_Ci_dict = defaultdict(lambda: {Ci: 0 for Ci in self.Ci_list})
        p_j_x_dict      = defaultdict(lambda: defaultdict(lambda: 0))
        p_j_Ci_x_dict   = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        # j : current demension of feature
        # Ci: class label
        for feature, label in zip(X, y):
            Ci = label
            for j, x in enumerate(feature):
                p_j_x_dict[j][x]        += 1
                p_j_Ci_x_dict[j][Ci][x] += 1
                count_j_dict[j] += 1
                count_j_Ci_dict[j][Ci] += 1

        # Calc average for p_j_x_dict
        for j in p_j_x_dict.keys():
            for x in p_j_x_dict[j].keys():
                p_j_x_dict[j][x] /= count_j_dict[j]

        # Calc average for p_j_Ci_x_dict
        for j in p_j_Ci_x_dict.keys():
            for Ci in p_j_Ci_x_dict[j].keys():
                for x in p_j_Ci_x_dict[j][Ci]:
                    p_j_Ci_x_dict[j][Ci][x] /= count_j_Ci_dict[j][Ci]


        self.p_j_x_dict    = p_j_x_dict
        self.p_j_Ci_x_dict = p_j_Ci_x_dict


    def _predict(self, X):
        _preds = []
        for feature in X:
            _pred = self._bayesian_map_pred(
                feature=feature
            )
            _preds.append(_pred)
        _preds = np.array(_preds)
        return _preds

    def predict(self, X):
        preds = []
        for _pred in self._predict(X):
            idx  = np.argmax(_pred)
            pred = self.Ci_list[idx]
            preds.append(pred)
        return preds

    def _bayesian_map_pred(self, feature):
        def _1(Ci, j):
            x = feature[j]
            numerator   = self.p_j_Ci_x_dict[j][Ci][x]
            denominator = self.p_j_x_dict[j][x]
            return 0 if denominator == 0 else numerator / denominator # TODO 0 is OK?

        return list(map(
            lambda Ci:
            np.product(list(map(lambda j: _1(Ci, j), range(0, len(feature))))),
            self.Ci_list
        ))
