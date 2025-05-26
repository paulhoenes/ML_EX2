import numpy as np

class NeuralNetwork:

    def __init__(self, input_size=8,
                 hidden1_size=100,
                 hidden2_size=50,
                 output_size=1,
                 lr=0.1,
                 activation_hidden='relu',
                 l2_lambda=0.0,
                 multiclass=False):

        # activation function
        self.activation_hidden = activation_hidden
        self.softmax_output = multiclass

        # Learning rate
        self.lr = lr
        # Weights and biases
        # Input -> Hidden1
        self.w0 = np.random.randn(hidden1_size, input_size) * 0.01
        self.b0 = np.zeros((hidden1_size, 1))

        # # Hidden1 → hidden2
        self.w1 = np.random.randn(hidden2_size, hidden1_size) * 0.01
        self.b1 = np.zeros((hidden2_size, 1))

        # Hidden2 -> Output
        self.w2 = np.random.randn(output_size, hidden2_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

        # Activation values
        self.a0 = None
        self.a1 = None
        self.a2 = None

        # L2 regularization parameter
        self.l2_lambda = l2_lambda

    def sigmoid_deriv(self, a):
        return a * (1 - a)

        # Activation function selector

    # Sigmoid for the output layer
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        # return expit(x)  # optional

    def activation(self, x):
        if self.activation_hidden == 'relu':
            return np.maximum(0, x)
        elif self.activation_hidden == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_hidden == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function")

    # Activation derivative selector
    def activation_deriv(self, a):
        if self.activation_hidden == 'relu':
            return (a > 0).astype(float)

        elif self.activation_hidden == 'sigmoid':
            return a * (1 - a)

        elif self.activation_hidden == 'tanh':
            return 1 - a ** 2

        else:
            raise ValueError("Unsupported activation function")


    # Loss function
    # Binary Cross-Entropy or softmax
    def loss(self, y_true, y_pred, l2_lambda=0.0):
        epsilon = 1e-12  # avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # if y_pred is < epsilon
        if self.softmax_output:
            loss_bce = -np.mean(np.sum(y_true * np.log(y_pred), axis= 0))
        else:
            loss_bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # L2 regularization
        l2_term = (
                np.sum(np.square(self.w0)) +
                np.sum(np.square(self.w1)) +
                np.sum(np.square(self.w2))
        )

        return loss_bce + l2_lambda * l2_term

    def loss_deriv(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis= 0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)


    def forward(self, X):

        self.X = X.T  # X.T: (input_size, m)

        ### 1. Hidden Layer
        # z0 = W0 @ X_T + b0
        # a0 = sig(z0)
        self.a0 = self.activation(self.w0 @ X.T + self.b0)

        ### 2. hidden layer
        # z1 = W1 @ a0 + b1
        # a1 = sig(z1)
        self.a1 = self.activation(self.w1 @ self.a0 + self.b1)

        # output Layer
        if self.softmax_output:
            # For softmax output, we use the exponential function
            self.a2 = self.softmax(self.w2 @ self.a1 + self.b2)
        else:
            self.a2 = self.sigmoid(self.w2 @ self.a1 + self.b2)

        return self.a2

    # Backward pass
    def backward(self, X, y_true):

        Y = y_true.T  # (output_size, m)
        m = Y.shape[1]  # samples

        y_true = y_true.T

        ### OUTPUT LAYER
        # delta2 = self.a1 - y_true.T # optional this simple calculation
        delta2 = self.loss_deriv(Y, self.a2) * self.sigmoid_deriv(self.a2)

        # Gradients for w2, b2:
        dw2 = (delta2 @ self.a1.T) / m
        db2 = np.sum(delta2, axis=1, keepdims=True) / m

        ### HIDDEN LAYER 2
        # (weights output layer x error output layer) * derived activation
        delta1 = (self.w2.T @ delta2) * self.activation_deriv(self.a1)

        # Gradients for w1, b1:
        # dw1: error in hidden layer 1 * outputs from hiddenlayer 2 (a2)
        dw1 = delta1 @ self.a0.T / m
        db1 = np.sum(delta1, axis=1, keepdims=True) / m  # sum errors in current layer

        ### HIDDEN LAYER 1
        # Error back to the previous layer
        delta0 = (self.w1.T @ delta1) * self.activation_deriv(self.a0)

        # Gradients for w0, b0:
        dw0 = (delta0 @ self.X.T) / m
        db0 = np.sum(delta0, axis=1, keepdims=True) / m

        # Update
        self.w2 -= self.lr * (dw2 + self.l2_lambda * self.w2)
        self.b2 -= self.lr * db2

        self.w1 -= self.lr * (dw1 + self.l2_lambda * self.w1)
        self.b1 -= self.lr * db1

        self.w0 -= self.lr * (dw0 + self.l2_lambda * self.w0)
        self.b0 -= self.lr * db0

    def train(self, X, y):
        # Get the prediction for current weights
        y_pred = self.forward(X)


        # Compute the loss
        current_loss = self.loss(y.T, y_pred)

        if np.isnan(current_loss):
            raise ValueError("Loss is NaN, check your data or model parameters.")

        # Update the weights and biases
        self.backward(X, y)
        return current_loss

    def predict(self, X):
        a1 = self.forward(X)
        return a1.T
