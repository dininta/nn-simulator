import pandas as pd
from grid_search import GridSearch
from mlp import MLPClassifier
from util import plot_learning_curve, evaluate_score, one_hot_encoding
import time


# Read data

file_path = 'data/monks-1.train'
data = pd.read_csv(file_path, sep='\s+', header=None)
y = data[0].to_numpy()
X = data[[1,2,3,4,5,6]]
X = one_hot_encoding(X)

n_attribute = X.shape[1]
n_output_unit = 1

#################################################

model = MLPClassifier(n_attribute, n_output_unit, shuffle=True, max_iter=250, batch_size=1)

#################################################

# Searching the best hyperparameter

#hyperparameters = {
#    'learning_rate': [0.01, 0.05, 0.1],
#    'n_unit_per_hidden_layer': [[7]],
#    'momentum': [0.4, 0.5, 0.6, 0.7],
#    'l2': [0.00005, 0.0001, 0.0002, 0.0003],
#    'hidden_activation_function': ['relu'],     # for hidden layer
#}
#
#grid_search = GridSearch(model, hyperparameters)
#grid_search.train(model, X, y, 5, 'mse')
#best_param, scores = grid_search.get_best_param()
#print("Best parameters:", best_param)

#################################################

# Train final model and plot the learning curve

best_param = {
    'learning_rate': 0.05,
    'n_unit_per_hidden_layer': [4],
    'momentum': 0.4,
    'l2': 0,
    'hidden_activation_function': 'relu',     # for hidden layer
}
model.set_params(**best_param)

file_path = 'data/monks-1.test'
data = pd.read_csv(file_path, sep='\s+', header=None)
y_test = data[0].to_numpy()
X_test = data[[1,2,3,4,5,6]]
X_test = one_hot_encoding(X_test)

start_time = time.time()
tr_scores, ts_scores = model.train(X, y, X_test, y_test, 'mse')
print("--- %s seconds ---" % (time.time() - start_time))

print('MSE (TR): ', evaluate_score('mse', y, model.predict(X)))
print('Acc (TR): ', evaluate_score('accuracy', y, model.predict(X)))
print('MSE (TS): ', evaluate_score('mse', y_test, model.predict(X_test)))
print('Acc (TS): ', evaluate_score('accuracy', y_test, model.predict(X_test)))

plot_learning_curve('MSE', (tr_scores, 'Training Set'), (ts_scores, 'Test Set'))
