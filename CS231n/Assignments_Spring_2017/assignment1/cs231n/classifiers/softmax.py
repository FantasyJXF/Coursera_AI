import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dW_each = np.zeros_like(dW)
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = np.dot(X, W)
  # # To invent the exponential value too much, substract the max(scores)
  scores_max = np.max(scores, axis=1)
  scores_max = np.reshape(scores_max, (num_train,1))
  prob = np.exp(scores - scores_max) / np.sum(np.exp(scores - scores_max), axis=1, keepdims=True)
  scores_corr = np.zeros_like(prob)
  scores_corr[np.arange(num_train), y] = 1.0
  for i in xrange(num_train):
      for j in xrange(num_class):  
          # 损失函数的公式L = -(1/N)∑i∑j1(k=yi)log(exp(fk)/∑j exp(fj)) + λR(W)  
          loss += -(scores_corr[i, j] * np.log(prob[i, j]))    
          #梯度的公式 ∇Wk L = -(1/N)∑i xiT(pi,m-Pm) + 2λWk, where Pk = exp(fk)/∑j exp(fj
          dW_each[:, j] = -(scores_corr[i, j] - prob[i, j]) * X[i, :]
      dW += dW_each
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot(X, W)  # N by C
  scores_max = np.max(scores, axis=1)
  scores_max = np.reshape(scores_max, (num_train,1))
  prob = np.exp(scores - scores_max) / np.sum(np.exp(scores - scores_max), axis=1, keepdims=True)
  scores_corr = np.zeros_like(prob)  # N by C
  scores_corr[np.arange(num_train), y] = 1.0  # labels是onehot类型,仅在正确样本处取1
  
  # 计算损失  scores_corr*C维度  np.log(prob)也是N*C的维度
  # loss = -log(scores) = -y_i + log(sum(exp(scores)))
  loss += -np.sum(scores_corr * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)

  # 计算损失  X.T = (D*N)  scores_corr-prob = (N*C)
  dW += -np.dot(X.T, scores_corr - prob) / num_train + reg * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

