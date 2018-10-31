
"""
code is adapted from: http://cs231n.stanford.edu/
"""

from builtins import range
from builtins import object
import numpy as np

from layers import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = np.random.normal(0,weight_scale,(input_dim,hidden_dim))
        self.params['b1'] = np.zeros((hidden_dim,))#np.random.normal(0,weight_scale,(hidden_dim,))
        self.params['W2'] = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
        self.params['b2'] = np.zeros((num_classes,))#np.random.normal(0,weight_scale,(num_classes,))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

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
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        H_x,cache = affine_forward(X,self.params["W1"],self.params["b1"])
        H_o,cache = relu_forward(H_x)
        Out_x,cache = affine_forward(H_o,self.params["W2"],self.params["b2"])
        scores = Out_x
        ''' 
        temp = 1/np.sum(np.exp(Out_x),axis = 1)
        C = len(self.params["b2"])
        temp = np.tile(temp.reshape(len(temp),1),(1,C))
        scores = np.exp(Out_x)*temp
        '''

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
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss,dx = softmax_loss(Out_x,y)
        loss += self.reg * 0.5 * np.linalg.norm(self.params["W1"]) ** 2
        loss += self.reg * 0.5 * np.linalg.norm(self.params["W2"]) ** 2
        """
        for k in self.params:
            param = self.params[k]
            loss += self.reg*0.5*np.linalg.norm(param)**2
        """
        dx,dw,db = affine_backward(dx,(H_o,self.params["W2"],self.params["b2"]))
        grads["W2"] = dw + self.params["W2"]*self.reg
        grads["b2"] = db #+ self.params["b2"]*self.reg
        dx = relu_backward(dx,H_x)
        dx,dw,db = affine_backward(dx,(X,self.params["W1"],self.params["b1"]))
        grads["W1"] = dw +self.params["W1"]*self.reg
        grads["b1"] = db #+self.params["b1"]*self.reg


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
