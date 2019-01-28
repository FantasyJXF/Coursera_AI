from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


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
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        pass
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
        X = X.reshape(X.shape[0], -1)
        a2, c1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, c2 = affine_forward(a2, self.params['W2'], self.params['b2'])
        pass
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
        loss_without_reg, dscores = softmax_loss(scores, y)
        loss = loss_without_reg + 0.5 * self.reg * (np.sum(self.params['W1']**2) + \
                                                    np.sum(self.params['W2']**2))
        da2, grads['W2'], grads['b2'] = affine_backward(dscores, c2)
        grads['W2'] += self.reg * c2[1]
        dx, grads['W1'], grads['b1'] = affine_relu_backward(da2, c1)
        grads['W1'] += self.reg * c1[0][1]
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        #初始化（L-1）个隐藏层的权值、偏置、batchnor
        for i in range(len(hidden_dims)):
            if 0 == i:
                dim_prev = input_dim
                dim_post = hidden_dims[i]
            else:
                dim_prev = hidden_dims[i-1]
                dim_post = hidden_dims[i]
            self.params['W'+str(i+1)] = weight_scale * np.random.randn(dim_prev, dim_post)
            self.params['b'+str(i+1)] = np.zeros(dim_post)      
            
            if self.use_batchnorm:
                self.params['gamma'+str(i+1)] = np.ones(dim_post)
                self.params['beta'+str(i+1)] = np.zeros(dim_post)

        #初始化输出层的权值、偏置
        self.params['W'+str(self.num_layers)] = weight_scale * np.random.randn(dim_post, num_classes)
        self.params['b'+str(self.num_layers)] = np.zeros(num_classes)      
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        #计算scores
        num_hiddenlayers = self.num_layers - 1
        caches = {} #记录各层每一级的cache，如第一层：“affine:cache11,batchnorm:cache12....."
        z={} #记录各层每一级的输出结果，如第一层：“affine:z11,batchnorm:z12....."
        #初始化网络输入值
        stringz_last="input"
        z[stringz_last] = X
        for i in range(num_hiddenlayers):
            # 当前层的第一个步骤: affine_forward
            stringz = "z" + str(i+1) + str(1)
            stringc = "cache" + str(i+1) + str(1)
            z[stringz], caches[stringc] = affine_forward(z[stringz_last], self.params['W'+str(i+1)], self.params['b'+str(i+1)])
            stringz_last = stringz

            # 当前层的第二个步骤[Optional]: batchnorm_forward
            if self.use_batchnorm:
                gamma = self.params['gamma' + str(i+1)]
                beta = self.params['beta' + str(i+1)]
                stringz = 'z' + str(i+1) + str(2)
                stringc = 'cache' + str(i+1) + str(2)
                z[stringz], caches[stringc] = batchnorm_forward(z[stringz_last], gamma, beta, self.bn_params[i])
                stringz_last = stringz
            
            # 当前层的第三个步骤: relu_forward
            stringz = "z" + str(i+1) + str(3)
            stringc = "cache" + str(i+1) + str(3)
            z[stringz], caches[stringc] = relu_forward(z[stringz_last])
            stringz_last = stringz

            # 当前层的第四个步骤[Optional]: dropout_forward
            if self.use_dropout:
                stringz = "z" + str(i+1) + str(4)
                stringc = "cache" + str(i+1) + str(4)
                z[stringz], caches[stringc] = dropout_forward(z[stringz_last], self.dropout_param)
                stringz_last = stringz
        #循环隐藏层结束

        #最后一个隐藏层，affine_forward + softmax_loss：
        stringc = 'cache' + '_out'
        stringz = 'z' + '_out'
        z[stringz], caches[stringc] = affine_forward(z[stringz_last], 
                                                     self.params['W' + str(self.num_layers)], 
                                                     self.params['b' + str(self.num_layers)])
        scores = z['z_out']
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #反向传播
        dz = {} #记录反向传播的中间值
        #计算loss
        loss_without_reg, dz['dz_out'] = softmax_loss(scores, y)
    
        for i in range(self.num_layers): 
            loss += 0.5* np.sum(np.square(self.params['W' + str(i+1)]))
        loss = loss + loss_without_reg
        
        #计算grads
        #最后一层affine反向传播并更新梯度
        stringW = 'W' + str(self.num_layers)
        stringb = 'b' + str(self.num_layers)
        stringdz = 'dz' + str(self.num_layers) + str(4)
        dz[stringdz], grads[stringW], grads[stringb] = affine_backward(dz['dz_out'], caches['cache_out'])
        grads[stringW] += self.reg * caches['cache_out'][1]
        stringdz_last = stringdz

        #（L-1)层隐藏层反向传播并更新梯度
        for i in range(num_hiddenlayers, 0, -1): 
            #dropout级反向
            if self.use_dropout:
                stringdz = 'dz' + str(i) + str(3)
                stringc = 'cache' + str(i) + str(4)
                dz[stringdz] = dropout_backward(dz[stringdz_last], caches[stringc])
                stringdz_last = stringdz
            #relu级反向
            stringdz = 'dz' + str(i) + str(2)
            stringc = 'cache' + str(i) + str(3)
            if not stringc in caches:
                stringc = 'cache' + str(i) + str(4)
            dz[stringdz] = relu_backward(dz[stringdz_last], caches[stringc])
            stringdz_last = stringdz
            #batchnorm级反向并更新梯度
            if self.use_batchnorm:
                stringdz = 'dz' + str(i) + str(1)
                stringc = 'cache' + str(i) + str(2)
                stringg = 'gamma' + str(i)
                stringbe = 'beta' + str(i)
                dz[stringdz],grads[stringg],grads[stringbe] = batchnorm_backward(dz[stringdz_last], caches[stringc])
                stringdz_last = stringdz
            #affine级反向并更新梯度
            stringdz = 'dz' + str(i) + str(4)
            stringc = 'cache' + str(i) + str(1)
            if not stringc in caches:
                stringc = 'cache' + str(i) + str(2)
            stringW = 'W' + str(i)
            stringb = 'b' + str(i)
            dz[stringdz],grads[stringW], grads[stringb] = affine_backward(dz[stringdz_last], caches[stringc])
            grads[stringW] += self.reg * caches[stringc][1]
            stringdz_last = stringdz
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
