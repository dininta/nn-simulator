import numpy as np
from util import *

class MLPBase:
    def __init__(self, n_attribute, n_output_unit, hidden_activation_function='relu', n_unit_per_hidden_layer=[], learning_rate=0.05, momentum=0, max_iter=400, batch_size=1, shuffle=False, seed=None, l2=0):
        self.n_attribute = n_attribute
        self.n_output_unit = n_output_unit
        self.n_unit_per_hidden_layer = n_unit_per_hidden_layer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.n_layer = len(n_unit_per_hidden_layer) + 1 # hidden layers and output layer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.hidden_activation_function = hidden_activation_function
        self.l2 = l2

    def get_max_iter(self):
        return self.max_iter

    def get_batch_size(self):
        return self.batch_size

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        self.n_layer = len(self.n_unit_per_hidden_layer) + 1 # hidden layers and output layer

    def is_output_layer(self, layer_idx):
        return layer_idx == self.n_layer - 1

    def compute_out(self, net, layer_idx):
        function_name = self.out_activation_function if self.is_output_layer(layer_idx) else self.hidden_activation_function
        f = activation_function(function_name)
        return f(net)

    def forward(self, x):
        net_per_layer = []
        out_per_layer = []

        inp = np.append(x, 1) # append bias
        for i in range(self.n_layer):
            net = np.matmul(self.weights[i], inp)
            out = self.compute_out(net, i)
            net_per_layer.append(net)
            out_per_layer.append(out)
            inp = np.append(out, 1)

        return net_per_layer, out_per_layer

    def backward(self, x, y, net_per_layer, out_per_layer):
        minus_gradient = []

        # Iterate from last layer
        for l in range(self.n_layer-1,-1,-1):
            current_layer_weight = self.weights[l]

            if self.is_output_layer(l):
                f = activation_function_der(self.out_activation_function)
                # e = (dⱼ - fⱼ(netⱼ))
                e = np.subtract(y, out_per_layer[l])
                # 𝛿ⱼ = fⱼ'(netⱼ) (dⱼ - fⱼ(netⱼ))
                d = np.multiply(f(net_per_layer[l]), e)
            else:
                f = activation_function_der(self.hidden_activation_function)
                # Σ𝑘 𝛿𝑘 w𝑘ⱼ
                sigma = np.matmul(d_previous_iter, self.weights[l+1][:,:-1])
                # 𝛿ⱼ = fⱼ'(netⱼ) Σ𝑘 𝛿𝑘 w𝑘ⱼ
                d = np.multiply(f(net_per_layer[l]), sigma)

            # keep 𝛿ⱼ for the next iteration
            d_previous_iter = d

            # input of current layer is output from previous layer
            inp = np.append(x, 1) if l == 0 else np.append(out_per_layer[l-1], 1)

            # - ∂E(w) / ∂wⱼᵢ = 𝛿ⱼ outᵢ
            minus_gradient.append(np.outer(d, inp))

        return minus_gradient[::-1]

    def train(self, X_train, y_train, X_test=None, y_test=None, scoring='mse'):
        self.weights = init_weight(self.n_unit_per_hidden_layer, self.n_attribute, self.n_output_unit)
        train_scores_per_epoch = []
        test_scores_per_epoch = []

        indexes = np.arange(X_train.shape[0])
        if self.seed is not None:
            np.random.seed(self.seed)

        for k in range(self.max_iter):
            previous_weight_delta = [None] * len(self.weights)

            n_batch = math.ceil(X_train.shape[0] / self.batch_size)

            # Shuffle training data for every epoch
            if self.shuffle:
                np.random.shuffle(indexes)

            for batch in range(n_batch):
                # Get data for current batch
                start_idx = batch * self.batch_size
                stop_idx = start_idx + self.batch_size
                batch_idx = indexes[start_idx:stop_idx]
                X = np.take(X_train, batch_idx, 0)
                y = np.take(y_train, batch_idx, 0)

                # Compute Δw per batch
                dw_per_batch = []
                for x, target in zip(X, y):
                    net_per_layer, out_per_layer = self.forward(x)
                    minus_gradient = self.backward(x, target, net_per_layer, out_per_layer)
                    dw_per_batch = minus_gradient if not dw_per_batch else [ np.add(d1, d2) for (d1, d2) in zip(dw_per_batch, minus_gradient) ]

                # Update weight after backpropagate 1 batch

                # Δwⱼᵢ = η [(1-α) 𝛿ⱼ outᵢ + α Δwⱼᵢ'] - λ wⱼᵢ
                for i in range(len(self.weights)):
                    dw_per_batch[i] = (1 - self.momentum) * dw_per_batch[i] / X.shape[0]
                    if previous_weight_delta[i] is not None:
                        previous_weight_delta[i] *= self.momentum 
                        dw_per_batch[i] += previous_weight_delta[i]
                    dw_per_batch[i] = (self.learning_rate * dw_per_batch[i]) - (self.weights[i] * self.l2)

                # w = w + Δw
                self.weights = np.add(self.weights, dw_per_batch)

                # Keep Δw for next iteration
                previous_weight_delta = dw_per_batch

            # Compute loss/accuracy per epoch
            if (X_test is not None):
                train_scores_per_epoch.append(evaluate_score(scoring, y_train, self.predict(X_train)))
                test_scores_per_epoch.append(evaluate_score(scoring, y_test, self.predict(X_test)))

        return train_scores_per_epoch, test_scores_per_epoch


class MLPClassifier(MLPBase):
    def __init__(self, n_attribute, n_output_unit, hidden_activation_function='relu', n_unit_per_hidden_layer=[], learning_rate=0.05, momentum=0, max_iter=400, batch_size=1, shuffle=False, seed=0, l2=0):
        super().__init__(n_attribute, n_output_unit, hidden_activation_function, 
            n_unit_per_hidden_layer, learning_rate, momentum, max_iter, batch_size, shuffle, seed, l2)
        self.out_activation_function = 'sigm' # use sigmoid for binary classification

    def predict(self, X):
        prediction = []
        for x in X:
            net_per_layer, out_per_layer = self.forward(x)
            output = out_per_layer[-1]
            if (output > 0.5):
                prediction.append(1)
            else:
                prediction.append(0)
        return prediction


class MLPRegressor(MLPBase):
    def __init__(self, n_attribute, n_output_unit, hidden_activation_function='relu', n_unit_per_hidden_layer=[], learning_rate=0.05, momentum=0, max_iter=400, batch_size=1, shuffle=False, seed=0, l2=0):
        super().__init__(n_attribute, n_output_unit, hidden_activation_function, 
            n_unit_per_hidden_layer, learning_rate, momentum, max_iter, batch_size, shuffle, seed, l2)
        self.out_activation_function = 'linear'

    def predict(self, X):
        prediction = []
        for x in X:
            net_per_layer, out_per_layer = self.forward(x)
            output = out_per_layer[-1]
            prediction.append(output)
        return prediction
