import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization,
    Param, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        _, _, n_channels = input_shape

        self.layers = []

        self.layers.append(ConvolutionalLayer(n_channels, conv1_channels, 3, 1))
        self.layers.append(ReLULayer())
        # self.layers.append(MaxPoolingLayer(4, 4))
        # self.layers.append(MaxPoolingLayer(4, 4))
        self.layers.append(MaxPoolingLayer(2, 2))

        self.layers.append(ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1))
        self.layers.append(ReLULayer())
        # self.layers.append(MaxPoolingLayer(4, 4))
        # self.layers.append(MaxPoolingLayer(4, 1))
        self.layers.append(MaxPoolingLayer(2, 2))

        self.layers.append(Flattener())
        # self.layers.append(FullyConnectedLayer(2*2*conv2_channels, n_output_classes))
        # self.layers.append(FullyConnectedLayer(5*5*conv2_channels, n_output_classes))
        self.layers.append(FullyConnectedLayer(8*8*conv2_channels, n_output_classes))

    def forward(self, X):
        x = X.copy()
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out):
        grad = d_out.copy()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        params = self.params()
        for p_key in params:
            param = params[p_key]
            param = Param(param.value)

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        x = self.forward(X)
        loss, grad = softmax_with_cross_entropy(x, y)
        grad = self.backward(grad)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = np.zeros(X.shape[0], np.int)
        x = self.forward(X)
        pred = np.argmax(softmax(x), axis=-1)

        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for it, layer in enumerate(self.layers):
            for name, value in layer.params().items():
                result[f'{it}_{name}'] = value

        return result
