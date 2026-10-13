from builtins import range
from builtins import object
from functools import cache

import os
import numpy as np

from softmax import *
from layers import *


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # Replace "pass" statement with your code
    out = x.reshape(x.shape[0], -1).dot(w) + b
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # Replace "pass" statement with your code
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # Replace "pass" statement with your code
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # Replace "pass" statement with your code
    dx = dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        normalization=None,
        dropout_keep_ratio=1,
        seed=None,
        dtype=np.float64
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.dtype = dtype
        self.params = {}
        self.reg = reg
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        # apply batchnorm, layernorm
        ############################################################################
        # Replace "pass" statement with your code
        self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params["b1"] = np.zeros(hidden_dim)

        self.params["W2"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b2"] = np.zeros(num_classes)

        if self.normalization == "batchnorm" or self.normalization == "layernorm":
            self.params["gamma1"] = np.ones(hidden_dim)
            self.params["beta1"] = np.zeros(hidden_dim)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed
        
        if self.normalization == "batchnorm":
            self.bn_param = {"mode": "train"}
        if self.normalization == "layernorm":
            self.ln_param = {}

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        # if self.use_dropout:
        #     self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            self.bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        # apply batchnorm, layernorm, dropout
        ############################################################################
        # Replace "pass" statement with your code
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        a1, fc_cache1 = affine_forward(X, W1, b1)

        if self.normalization == "batchnorm":
            a1, norm_cache = batchnorm_forward(
                a1, self.params["gamma1"], self.params["beta1"], self.bn_param
            )
        elif self.normalization == "layernorm":
            a1, norm_cache = layernorm_forward(
                a1, self.params["gamma1"], self.params["beta1"], self.ln_param
            )

        h1, relu_cache = relu_forward(a1)

        if self.use_dropout:
            self.dropout_param["mode"] = mode
            h1, dropout_cache = dropout_forward(h1, self.dropout_param)

        scores, fc_cache2 = affine_forward(h1, W2, b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # apply batchnorm, layernorm, dropout
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # Replace "pass" statement with your code
        # Convert scores to probabilities
        prob, softmax_cache = softmax_forward(scores)
        
        # Calculate cross-entropy loss
        loss, ce_cache = cross_entropy_forward(prob, y)
        
        # Backward pass through cross-entropy
        dprob = cross_entropy_backward(ce_cache)
        
        # Backward pass through softmax
        dscores = softmax_backward(dprob, softmax_cache)

        loss += 0.5 * self.reg * (
            np.sum(W1 * W1) + np.sum(W2 * W2)
        )

        dh1, dW2, db2 = affine_backward(dscores, fc_cache2)

        if self.use_dropout:
            dh1 = dropout_backward(dh1, dropout_cache)

        dh1 = relu_backward(dh1, relu_cache)

        if self.normalization == "batchnorm":
            dh1, dgamma1, dbeta1 = batchnorm_backward(dh1, norm_cache)
            grads["gamma1"] = dgamma1
            grads["beta1"] = dbeta1
        elif self.normalization == "layernorm":
            dh1, dgamma1, dbeta1 = layernorm_backward(dh1, norm_cache)
            grads["gamma1"] = dgamma1
            grads["beta1"] = dbeta1

        dX, dW1, db1 = affine_backward(dh1, fc_cache1)

        grads["W1"] = dW1 + self.reg * W1
        grads["b1"] = db1
        grads["W2"] = dW2 + self.reg * W2
        grads["b2"] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
