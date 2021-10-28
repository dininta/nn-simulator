import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Activation Functions

def linear(x):
    return x

def heaviside(x):
    return 1 if x>0 else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #float to avoid int_to_float conversion

def tanh(x):
    return math.tanh(x)

def relu(x):
    return max(0, x)

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def leakyrelu(x, a=0.01):
    """
    Leaky version of a Rectified Linear Unit.
    It allows a small gradient when the unit is not active:
    f(x) = alpha * x if x < 0
    f(x) = x if x >= 0
    """
    return x*a if x < 0 else x

functions = {
    'linear': linear,
    'heav': heaviside,
    'sigm': sigmoid,
    'tan': tanh,
    'relu': relu,
    'softp': softplus,
    'leakyr': leakyrelu
}

def activation_function(function_name):
    return np.vectorize(functions[function_name])


# Activation Functions Derivatives

def linear_der(x):
    return 1

def heaviside_der(x):
    return 0

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_der(x):
    return 1 - (math.tanh(x))**2

def relu_der(x):
    return np.greater(x, 0).astype(int)

def softplus_der(x):
    return 1. / (1. + np.exp(-x))

def leakyrelu_der(x, a=0.01):
    return a if x < 0 else 1

funct_der = {
    'linear': linear_der,
    'heav': heaviside_der,
    'sigm': sigmoid_der,
    'tan': tanh_der,
    'relu': relu_der,
    'softp': softplus_der,
    'leakyr': leakyrelu_der
}

def activation_function_der(function_name):
    return np.vectorize(funct_der[function_name])


# Evaluation Scores

def accuracy(target, prediction):
    """
    Evaluation score for binary classification
    Returns percentage of correctly classified data

    Input:
        - target: 1-d array of numeric
        - prediction: 1-d array of numeric
    Return:
        - scores: float in [0,1]
    """
    sum = 0
    for (t, p) in zip(target, prediction):
        sum += 1 if (t == p) else 0
    return sum / (len(target))

def mean_squared_error(target, prediction):
    """
    Evaluation score for classification & (single & multiple target) regression

          1                 1    1
    mse = - Σᵢ (tᵢ - oᵢ)² = - Σᵢ - ((tᵢ₁ - oᵢ₁)² + (tᵢ₂ - oᵢ₂)² + ... + (tᵢⱼ - oᵢⱼ)²)
          N                 N    j

    Input:
        - target: 1-d array of numeric or 2-d array of numeric
        - prediction: 1-d array of numeric or 2-d array of numeric
    Return:
        - scores: float
    """
    sum = 0
    for (t, p) in zip(target, prediction):
        sum += (t - p) ** 2
    return np.mean(sum / (len(target)))

def mean_euclidean_error(target, prediction):
    """
    Evaluation score for classification & (single & multiple target) regression
    For single value regression, will return mean absolute error

          1                 1
    mee = - Σᵢ ‖tᵢ - oᵢ‖₂ = - Σᵢ √((tᵢ₁ - oᵢ₁)² + (tᵢ₂ - oᵢ₂)² + ... + (tᵢⱼ - oᵢⱼ)²)
          N                 N

    Input:
        - target: 1-d array of numeric or 2-d array of numeric
        - prediction: 1-d array of numeric or 2-d array of numeric
    Return:
        - scores: float
    """
    sum = 0
    for (t, p) in zip(target, prediction):
        sum += math.sqrt(np.sum((t - p) ** 2))
    return sum / (len(target))

def evaluate_score(scoring, target, prediction):
    score_function = {
        'accuracy': accuracy,
        'mse': mean_squared_error,
        'mee': mean_euclidean_error,
    }[scoring]
    return score_function(target, prediction)


# Random Weight Initialization

def init_weight(n_unit_per_hidden_layer, n_attribute, n_output_unit):
    weights = []

    # Hidden layer
    n_hidden_layer = len(n_unit_per_hidden_layer)
    weight_length = n_attribute + 1
    for i in range(n_hidden_layer):
        n_unit = n_unit_per_hidden_layer[i]
        w = 1.4 * np.random.random_sample((n_unit, weight_length)) - 0.7  # weight in range [-0.7, 0.7]
        weights.append(w)
        weight_length = n_unit + 1

    # Output layer
    w = 1.4 * np.random.random_sample((n_output_unit, weight_length)) - 0.7  # weight in range [-0.7, 0.7]
    weights.append(w)

    return weights

def print_weight(weights):
    for i in range(len(weights)):
        print("Layer", i)
        current_layer_weight = weights[i]
        n_unit = len(current_layer_weight)
        print("Num unit:", n_unit)
        for j in range(n_unit):
            print(current_layer_weight[j])


# Cross Validation

def kfold_cross_validate(model, X, y, num_fold, scoring, shuffle=False, seed=None):
    indexes = np.arange(X.shape[0])
    if seed is not None:
        np.random.seed(seed)
    if shuffle:
        np.random.shuffle(indexes)

    folds = np.array_split(indexes, num_fold)
    train_scores = []
    val_scores = []

    for i in range(0, num_fold):
        train_index = np.concatenate((folds[:i] + folds[i+1:]))
        val_index = folds[i]

        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]

        # Train model
        model.train(X_train, y_train)

        # Calculate error
        val_prediction = model.predict(X_val)
        val_score = evaluate_score(scoring, y_val, val_prediction)
        val_scores.append(val_score)

        train_prediction = model.predict(X_train)
        train_score = evaluate_score(scoring, y_train, train_prediction)
        train_scores.append(train_score)

    return train_scores, val_scores


# Split Train and Test Set

def train_test_split(X, y, test_size=0.2, shuffle=False):
    indexes = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indexes)

    pivot = math.ceil(test_size*X.shape[0])
    test_idx = indexes[:pivot]
    train_idx = indexes[pivot:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# Plot Learning Curve

def plot_learning_curve(ylabel, *data_per_epoch, filename=None):
    """
    Sample of usage:
        plot_learning_curve('MSE', (train_scores, 'Training Scores'), (val_scores, 'Validation Scores'), filename='figure.png')
        plot_learning_curve('MEE', (scores_1, 'With regularization'), (scores_2, 'Without regularization'))
    Input:
        - ylabel: string
        - data_per_epoch: tuple of 1-d array of numeric and string
        - filename: string, if supplied, will save the figure
    """
    lines = ['','--',':','-.']
    i = 0;
    for data in data_per_epoch:
        x = np.arange(len(data[0])) * 1
        plt.plot(x, data[0], lines[i], label = data[1])
        i = (i+1) % len(lines)

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()

    if filename is not None:
        plt.savefig(filename)
    plt.clf()


# One Hot Encoding

def one_hot_encoding(x_data):
    input_nodes = []

    for row in range(x_data.shape[0]):
    
        temp_row = []
    
        for index, item in x_data.iloc[row].items():
    
            if (index == 1) or (index == 2) or (index == 4):
                if item == 1:
                    temp_row = np.hstack((temp_row, [1, 0, 0]))
                if item == 2:
                    temp_row = np.hstack((temp_row, [0, 1, 0]))
                if item == 3:
                    temp_row = np.hstack((temp_row, [0, 0, 1]))
        
            if (index == 3) or (index == 6):
                if item == 1:
                    temp_row = np.hstack((temp_row, [1, 0]))
                if item == 2:
                    temp_row = np.hstack((temp_row, [0, 1])) 
                    
            if index == 5:
                if item == 1:
                    temp_row = np.hstack((temp_row, [1, 0, 0, 0]))
                if item == 2:
                    temp_row = np.hstack((temp_row, [0, 1, 0, 0]))
                if item == 3:
                    temp_row = np.hstack((temp_row, [0, 0, 1, 0]))
                if item == 4:
                    temp_row = np.hstack((temp_row, [0, 0, 0, 1]))
                       
        if row == 0:
            input_nodes = temp_row # initialize
        else:
            input_nodes = np.vstack((input_nodes, temp_row)) # pile up    

    return input_nodes
