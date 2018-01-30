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

        p_j_x_dict_count    = 0
        p_j_x_Ci_count_dict = {Ci: 0 for Ci in self.Ci_list}
        p_j_x_dict          = defaultdict(lambda: defaultdict(lambda: 0.0))
        p_j_x_Ci_dict       = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        # j : current demension of feature
        # Ci: class label
        for feature, label in zip(X, y):
            Ci = label
            for j, x in enumerate(feature):
                p_j_x_dict[j][x]        += 1
                p_j_x_Ci_dict[j][x][Ci] += 1
                p_j_x_dict_count += 1
                p_j_x_Ci_count_dict[Ci] += 1

        # Calc average for p_j_x_dict
        for j in p_j_x_dict.keys():
            for x in p_j_x_dict[j].keys():
                p_j_x_dict[j][x] /= p_j_x_dict_count

        # Calc average for p_j_x_Ci_dict
        for j in p_j_x_Ci_dict.keys():
            for x in p_j_x_Ci_dict[j].keys():
                for Ci in self.Ci_list:
                    p_j_x_Ci_dict[j][x][Ci] /= p_j_x_Ci_count_dict[Ci]


        self.p_j_x_dict    = p_j_x_dict
        self.p_j_x_Ci_dict = p_j_x_Ci_dict


    def _predict(self, X):
        _preds = []
        for feature in X:
            _pred = self._bayesian_map_pred(
                p_j_x_Ci_dict=self.p_j_x_Ci_dict,
                p_j_x_dict=self.p_j_x_dict,
                Ci_list=self.Ci_list,
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

    def _bayesian_map_pred(self, p_j_x_Ci_dict, p_j_x_dict, Ci_list, feature):
        def _1(Ci, j):
            x = feature[j]
            numerator   = p_j_x_Ci_dict[j][x][Ci]
            denominator = p_j_x_dict[j][x]
            return 0 if denominator == 0 else numerator / denominator # TODO 0 is OK?

        return list(map(
            lambda Ci:
            np.product(list(map(lambda j: _1(Ci, j), range(0, len(feature))))),
            Ci_list
        ))
