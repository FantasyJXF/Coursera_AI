# -*- coding:utf-8 -*-
import numpy as np
from random import shuffle
from past.builtins import xrange

'''
svm算法就是对于每一个样本，计算其他不正确分类与正确分类之间的差距，
如果差距大于delta那就说明差距大，需要进行进一步优化，所以对于每个样本，
把这些差距加起来，让他们尽量小，这个就是svm的核心思想。
'''
def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]] # 计算得到第i个样本的真实分类的分数
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      # Multiclass SVM损失函数计算
      # Li = sum(max(0, sj - si + 1))  while  j!=yi
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:]
        dW[:,y[i]] += -X[i,:]  # y[i]是正确的类
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)        # N by C  样本数*类别数
  num_train = X.shape[0]
  #num_classes = W.shape[1]

  # 从scores的第i行中取出y[i]对应的正确标签的分数, i = np.arrange(num_train)
  scores_correct = scores[np.arange(num_train), y]    # 1*N 
  scores_correct = np.reshape(scores_correct, (num_train, 1))  # N*1 每个样本的正确类别

  margins = scores - scores_correct + 1.0     # N by C   计算scores矩阵中每一处的损失
  margins[np.arange(num_train), y] = 0.0      # 将每个样本的正确类别损失置为0
  margins[margins <= 0] = 0.0                 # max(0, x)
  loss += np.sum(margins) / num_train         # 累加所有损失，取平均
  loss += 0.5 * reg * np.sum(W * W)           # 正则
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margins[margins > 0] = 1.0                  # max(0, x)  大于0的梯度计为1
  row_sum = np.sum(margins, axis=1)           # N*1  每个样本累加
  margins[np.arange(num_train), y] = -row_sum  # 类正确的位置 = -梯度累加
  dW += np.dot(X.T, margins)/num_train + reg * W     # D by C
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
