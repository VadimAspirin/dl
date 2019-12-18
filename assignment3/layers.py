import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(W**2)
    grad = (reg_strength * W) * 2
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    norm_pred = predictions - np.max(predictions, axis=-1, keepdims=True)
    exp_pred = np.exp(norm_pred)
    return exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    q = np.log(probs)
    p = np.eye(probs.shape[-1])[target_index].reshape(probs.shape)
    return -np.sum(p*q) / np.count_nonzero(p)


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    softmax_pred = softmax(predictions)
    loss = cross_entropy_loss(softmax_pred, target_index)
    target_dped = np.eye(softmax_pred.shape[-1])[target_index].reshape(softmax_pred.shape)
    dprediction = softmax_pred - target_dped
    return loss, dprediction / np.count_nonzero(target_dped)


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X.copy()
        return np.where(X <= 0, 0, X)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops

        d_result = np.where(self.X <= 0, 0, 1) * d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.dot(np.ones((1, self.X.shape[0])), d_out)
        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.X = None
        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        out_height = 0
        out_width = 0
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        self.X = X.copy()
        if self.padding:
            # zero_columns = np.zeros((batch_size, height, self.padding, channels))
            # self.X = np.append(self.X, zero_columns, axis=2)
            # self.X = np.append(zero_columns, self.X, axis=2)
            # width = width + self.padding * 2
            # zero_rows = np.zeros((batch_size, self.padding, width, channels))
            # self.X = np.append(self.X, zero_rows, axis=1)
            # self.X = np.append(zero_rows, self.X, axis=1)
            self.X = np.pad(self.X, ((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
            batch_size, height, width, channels = self.X.shape


        out_height = (height - self.filter_size) + 1
        out_width = (width - self.filter_size) + 1
        result = np.zeros([batch_size, out_height, out_width, self.out_channels])

        W = self.W.value.reshape(-1, self.out_channels)

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                X_slice = self.X[:, x:self.filter_size+x, y:self.filter_size+y, :].reshape(batch_size, -1)
                result[:, x, y, :] = np.dot(X_slice, W) + self.B.value

        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        self.W.grad = np.zeros(self.W.value.shape)
        d_input = np.zeros(self.X.shape)
        self.B.grad = np.zeros(self.B.value.shape)

        W = self.W.value.reshape(-1, self.out_channels)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                X_slice = self.X[:, x:self.filter_size+x, y:self.filter_size+y, :].reshape(batch_size, -1)
                W_grad = np.dot(X_slice.T, d_out[:, x, y, :])
                W_grad = W_grad.reshape(self.filter_size, self.filter_size, channels, out_channels)
                self.W.grad += W_grad

                X_grad = np.dot(d_out[:, x, y, :], W.T)
                X_grad = X_grad.reshape(batch_size, self.filter_size, self.filter_size, channels)
                d_input[:, x:self.filter_size+x, y:self.filter_size+y, :] += X_grad

                B_grad = np.dot(np.ones((1, batch_size)), d_out[:, x, y, :])
                B_grad = B_grad.reshape(-1)
                self.B.grad += B_grad

        if self.padding:
            d_input = d_input[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X.copy()
        out_height = ((height - self.pool_size) // self.stride) + 1
        out_width = ((width - self.pool_size) // self.stride) + 1
        result = np.zeros([batch_size, out_height, out_width, channels])

        for y in range(out_height):
            for x in range(out_width):
                result[:, x, y, :] = np.max(self.X[:, x:self.pool_size+x, y:self.pool_size+y, :], axis=(1,2))

        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        d_input = np.zeros(self.X.shape)

        for y in range(out_height):
            for x in range(out_width):
                X_slice = self.X[:, x:self.pool_size+x, y:self.pool_size+y, :]
                d_out_it = d_out[:, x, y, :].copy()

                mask = np.where(X_slice == np.max(X_slice, axis=(1,2), keepdims=True), 1, 0)
                count_max = np.count_nonzero(mask, axis=(1,2))
                # d_out_it = d_out_it / count_max
                # d_out_it = np.float64(d_out_it) / np.float64(count_max)
                d_out_it = d_out_it.astype(np.float64) / count_max.astype(np.float64)

                d_out_it = d_out_it.reshape(batch_size, 1, 1, out_channels)
                d_input[:, x:self.pool_size+x, y:self.pool_size+y, :] += mask * d_out_it

        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        batch_size, height, width, channels = self.X_shape
        return d_out.reshape(batch_size, height, width, channels)

    def params(self):
        # No params!
        return {}
