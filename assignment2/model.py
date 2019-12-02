import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, Param, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu1 = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def forward(self, X):
        x = self.layer1.forward(X)
        x = self.relu1.forward(x)
        x = self.layer2.forward(x)
        return x

    def backward(self, d_out):
        grad = self.layer2.backward(d_out)
        grad = self.relu1.backward(grad)
        grad = self.layer1.backward(grad)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        params = self.params()
        for p_key in params:
            param = params[p_key]
            param = Param(param.value)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        x = self.forward(X)
        loss, grad = softmax_with_cross_entropy(x, y)
        grad = self.backward(grad)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        params = self.params()
        for p_key in params:
            param = params[p_key]
            l2_loss, l2_grad = l2_regularization(param.value, self.reg)
            loss += l2_loss
            param.grad += l2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        x = self.forward(X)
        pred = np.argmax(softmax(x), axis=-1)

        return pred

    def params(self):
        #result = {}
        result = {'layer1_W': self.layer1.params()['W'],
                  'layer2_W': self.layer2.params()['W']}

        return result
