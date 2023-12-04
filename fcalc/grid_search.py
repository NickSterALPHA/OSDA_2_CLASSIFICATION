import numpy as np

from . import patterns
from . import binary_decision_functions
from . import classifier


class GridSearch:
    def __init__(self, cv, range_param, model, score, X, y) -> None:
        self.cv = cv
        self.model = model
        self.range_param = range_param
        self.score = score
        self.X = X
        self.y = y
        self.best_params = 0
        self.max_score = 0

    def cross_val_score(self, param):
        first = last = 0
        num_rows = self.X.shape[0]
        len_blocks = num_rows // self.cv
        cur_block = 1
        arr_score = []

        while first < num_rows:
            if cur_block == self.cv:
                last = num_rows 
            else:
                last = first + len_blocks
            
            mask_array = np.full(shape=num_rows, fill_value=True, dtype=bool)
            mask_array[first:last] = False

            X_train = self.X[mask_array, :]
            y_train = self.y[mask_array]
            X_test = self.X[first:last, :]
            y_test = self.y[first:last]

            self.model.fit(X_train, y_train, param)
            y_pred = self.model.predict(X_test)

            arr_score.append(self.score(y_test, y_pred))
            
            first = last
            cur_block += 1

        cur_score = sum(arr_score) / len(arr_score)

        return cur_score
    
    def Search(self):
        start = self.range_param[0]
        end = self.range_param[1]
        step = self.range_param[2]

        for param in np.arange(start, end, step):
            score = self.cross_val_score(param)
            if self.max_score < score:
                self.max_score = score
                self.best_params = param