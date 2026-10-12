from __future__ import print_function

import os
from builtins import range
from builtins import object
import numpy as np
from softmax import *
from past.builtins import xrange




def softmax_forward(x):
    """
    Computes the softmax scores for each class given the logits.

    Inputs:
    - x: Input logits of shape (B, K) where x[i, j] is the logit for the jth class
      for the ith input.

    Returns:
    - out: Output data of shape (B, K) where out[i, j] is the probability for
      the jth class for the ith input.
    - cache: Input x, of same shape as out, which will be used for the backward pass.
    """
    out = None
    ###########################################################################
    # TODO:                                                                   #
    # Compute the softmax probabilities for x with shape of (B, K).
    # B is batch size, K is number of classes. Each row of the output should sum to 1. 
    # The output should have shape (B, K)         #
    ###########################################################################    
    # Replace "pass" statement with your code
    pass
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    cache = x
    return out, cache


def softmax_backward(dout, cache):
    """
    Computes the backward pass for the softmax function.

    Inputs:
    - dout: Upstream derivatives, of shape (N, K)
    - cache: Input x, of same shape as dout

    Returns:
    - dL/dZ: Gradient with respect to z
    """
    """Accept dL/dS and return dL/dZ using dZ = dS @ J."""
    Z = cache
    dS = dout
    dZ = None
    ###########################################################################
    # TODO:                                                                   #
    # Compute the gradient of the loss with respect to the input logits Z.   #
    # dout is the upstream gradient dL/dS, with shape (B, K) where B is 
    # the batch size and K is the number of classes.  #
    # For each sample, construct the softmax Jacobian J of shape (K, K),      #
    # where J[i, j] = S[i] * (delta_ij - S[j]) and delta_ij is 1 if i == j,   #
    # and 0 otherwise. Apply the chain rule by multiplying the upstream      #
    # gradient for that sample by J. 
    # The dZ is a batched gradient of shape (B, K).
    # hint: to get batched gradient, compute one sample at a time and 
    # then stack the results together using np.stack.
    ###########################################################################    
    # Replace "pass" statement with your code
    pass
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return dZ



def cross_entropy_forward(prob, y):
    """
    Computes the cross-entropy loss for a batch of predictions.

    Inputs:
    - prob: Input data of shape (B, K) where prob[i, j] is the probability for the jth class
      for the ith input.
    - y: ground truth labels of shape (B,) where y[i] is the label for X[i] and 0 <= y[i] < K
         or one-hot encoded labels of shape (B, K) where y[i, j] is 1 if the jth class is the correct label for the ith input, and 0 otherwise.
    Returns:
    - loss: Scalar giving the cross-entropy loss
    """
    # make sure y is one-hot encoded given y is labels like [0,3]
    if len(y.shape) == 1:
        y_onehot = np.eye(prob.shape[1])[y]
    else:
        y_onehot = y
    loss = None
    ###########################################################################
    # TODO:                                                                   #
    # Compute the mean cross-entropy loss and store the scalar in loss.
    # input is batched probabilities, prob has shape (B, K). and ground truth
    # labels y_onehot have shape (B, K).       #
    # Multiply the negative log probabilities by y_onehot, sum over all       #
    # samples and classes, and divide by the batch size B.     #
    # Store (prob, y_onehot) in cache for the backward pass.        #
    ###########################################################################
    # Replace "pass" statement with your code
    pass
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    cache = (prob, y_onehot)


    return loss, cache




def cross_entropy_backward(cache):
    """
    Computes the backward pass for the cross-entropy loss.

    Inputs:
    - cache: Tuple of (prob, y_onehot) where prob is the predicted probability distribution and y_onehot is the one-hot encoded labels

    Returns:
    - dS: Gradient with respect to prob
    """
    prob, y = cache
    # make sure y is one-hot encoded given y is labels like [0,3]
    if len(y.shape) == 1:
        y_onehot = np.eye(prob.shape[1])[y]
    else:
        y_onehot = y
    dS = None
    ###########################################################################
    # TODO:                                                                   #
    # Compute the gradient of the mean cross-entropy loss with respect to     #
    # the predicted probabilities, with shape (B, K).                 #
    ###########################################################################   

    # Replace "pass" statement with your code
    pass
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    
    
    
    return dS


