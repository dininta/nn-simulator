import itertools as it
import numpy as np
from util import kfold_cross_validate, plot_learning_curve
import time

class GridSearch:
    def __init__(self, model, hyperparams):
        self.model = model
        self.hyperparams = self.construct_params(hyperparams)
        self.result = []
        self.best_param_idx = 0

    def construct_params(self, hyperparams):
        keys = hyperparams.keys()
        values = (hyperparams[key] for key in keys)
        return [dict(zip(keys, combination)) for combination in it.product(*values)]

    def set_best_param(self, scoring):
        function = {
            'accuracy': max,
            'mse': min,
            'mee': min,
        }[scoring]
        seq = [x['VL_mean'] for x in self.result]
        self.best_param_idx = seq.index(function(seq))

    def get_best_param(self):
        return self.hyperparams[self.best_param_idx], self.result[self.best_param_idx]

    def train(self, model, X, y, num_fold, scoring):
        self.result = []
        for params in self.hyperparams:
            print('param', params)
            model.set_params(**params)

            start_time = time.time()
            train_scores, val_scores = kfold_cross_validate(model, X, y, num_fold, scoring, shuffle=True)
            train_scores_mean, val_scores_mean = np.mean(train_scores), np.mean(val_scores)
            self.result.append({
                'TR_scores': train_scores,
                'VL_scores': val_scores,
                'TR_mean': train_scores_mean,
                'VL_mean': val_scores_mean,
            })
            print("--- %s seconds ---" % (time.time() - start_time))

            print('TR_mean', train_scores_mean)
            print('VL_mean', val_scores_mean)

        self.set_best_param(scoring)
