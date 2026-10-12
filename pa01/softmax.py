from __future__ import print_function

import os
from builtins import range
from builtins import object
import numpy as np
from pa01.softmax import *
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
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted_x)
    out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
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
    shifted_Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(shifted_Z)
    S = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    # Compute one sample at a time
    gradients = []

    for i in range(Z.shape[0]):
        s = S[i]

        J = np.diag(s) - np.outer(s, s)

        # Chain rule:
        # dZ = dS @ J
        dz_i = dS[i].dot(J)

        gradients.append(dz_i)

    dZ = np.stack(gradients)
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
    eps = 1e-12

    # Clip probabilities for numerical stability
    prob_clipped = np.clip(prob, eps, 1.0)

    # Cross entropy:
    # L = -1/B * sum(y * log(prob))
    loss = -np.sum(y_onehot * np.log(prob_clipped)) / prob.shape[0]
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
    eps = 1e-12

    prob_clipped = np.clip(prob, eps, 1.0)

    dS = -y_onehot / prob_clipped / prob.shape[0]
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    
    
    
    return dS



class LinearClassifier(object):
    def __init__(self):
        self.W = None
        

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = 10
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)
        
        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,). 
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.  
            # 
            # compute forward pass of
            #   -linear transformation
            #   -softmax
            #   -cross-entropy loss
            #   - save the intermediate results in cache for backward pass

            # compute backward pass of
            #   -cross-entropy loss
            #   -softmax
            #   -linear transformation
            #   -get dW, the gradient of the loss with respect to W
            # 
            # regularization
            #   - add L2 regularization to the loss with reg as the regularization strength
            #   - add L2 regularization to the gradient dW with reg as the regularization strength
            # 
            # Gradient descent update
            #   - update W using the gradient dW and the learning rate 
            #########################################################################
            # Replace "pass" statement with your code
            batch_indices = np.random.choice(num_train, batch_size, replace = True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            scores = X_batch.dot(self.W)
            probs, softmax_cache = softmax_forward(scores)
            loss, ce_cache = cross_entropy_forward(probs,y_batch)
            loss += reg * np.sum(self.W * self.W)

            dprobs = cross_entropy_backward(ce_cache)
            dscores = softmax_backward(dprobs,softmax_cache)
            dW = X_batch.T.dot(dscores)
            dW += 2 * reg * self.W
            loss_history.append(loss)
            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            # Replace "pass" statement with your code
            self.W -= learning_rate * dW
            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # Replace "pass" statement with your code
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis = 1)
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        return y_pred

    
    
