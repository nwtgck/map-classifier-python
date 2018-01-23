# Bayesian Maximum a posteriori estimation (MAP)

from collections import defaultdict
import numpy as np

def _bayesian_map_pred(p_j_x_Ci_dict, x_j_dict, p_j_x_dict, j_list, Ci_list):
    def _1(Ci, j):
        x = x_j_dict[j]
        return p_j_x_Ci_dict[j][x][Ci] / p_j_x_dict[j][x]

    return list(map(
        lambda Ci:
        np.product(list(map(lambda j: _1(Ci, j), j_list))),
        Ci_list
    ))


def bayesian_map_pred(p_j_x_Ci_dict, x_j_dict, p_j_x_dict, j_list, Ci_list):
    _pred = _bayesian_map_pred(p_j_x_Ci_dict, x_j_dict, p_j_x_dict, j_list, Ci_list)
    return max(zip(_pred, Ci_list), key=lambda p_and_Ci: p_and_Ci[0])[1]


class MAPClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        assert (len(X) == len(y))

        self.p_j_x_dict    = defaultdict(lambda: defaultdict(lambda: 0))
        self.p_j_x_Ci_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.Ci_list       = sorted(list(set(y)))
        # j : current demension of feature
        # Ci: class label
        for feature, label in zip(X, y):
            Ci = label
            for j, x in enumerate(feature):
                self.p_j_x_dict[j][x] += 1
                self.p_j_x_Ci_dict[j][x][Ci] += 1

        print("FIT!")

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
            pred  = max(zip(_pred, self.Ci_list), key=lambda p_and_Ci: p_and_Ci[0])[1]
            preds.append(pred)
        return preds

    def _bayesian_map_pred(self, p_j_x_Ci_dict, p_j_x_dict, Ci_list, feature):
        def _1(Ci, j):
            x = feature[j]
            numerator   = p_j_x_Ci_dict[j][x][Ci]
            denominator = p_j_x_dict[j][x]
            # if denominator != 0:
            #     print("NOT ZERO~~~~~~~")
            # print("denominator:", denominator)
            # return numerator / denominator
            return 0 if denominator == 0 else numerator / denominator # TODO 0 is OK?

        return list(map(
            lambda Ci:
            np.product(list(map(lambda j: _1(Ci, j), range(0, len(feature))))),
            Ci_list
        ))
