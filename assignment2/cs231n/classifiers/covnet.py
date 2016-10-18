import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class CovNet(object):
    def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=3,
                 hidden_dim=128, num_classes=10, weight_scale=1e-3, reg=0.0,
                 N=2, M=2, dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        stride = 1
        pad = (filter_size - 1)/2
        C, H, W = input_dim
        for i in xrange(N):
            self.params['CW'+str(2*i+1)] = weight_scale*np.random.randn(num_filters, C, filter_size, filter_size)
            self.params['Cb'+str(2*i+1)] = np.zeros((num_filters,))
            Hc = (H + 2*pad - filter_size)/stride + 1
            Wc = (W + 2*pad - filter_size)/stride + 1
            self.params['Sgamma'+str(2*i+1)] = weight_scale*np.random.randn(num_filters)
            self.params['Sbeta'+str(2*i+1)]  = np.zeros((num_filters,))
            self.params['CW'+str(2*i+2)] = weight_scale*np.random.randn(num_filters, num_filters, filter_size, filter_size)
            self.params['Cb'+str(2*i+2)] = np.zeros((num_filters,))
            self.params['Sgamma'+str(2*i+2)] = weight_scale*np.random.randn((num_filters,))
            self.params['Sbeta'+str(2*i+2)]  = np.zeros((num_filters,))
            Hc = (Hc + 2*pad - filter_size)/stride + 1
            Wc = (Wc + 2*pad - filter_size)/stride + 1
            # for max pooling pool_height=pool_with=2, strid=2
            Ha = (Hc - 2)/2 + 1
            Wa = (Wc - 2)/2 + 1


        self.params['FCW1'] = weight_scale*np.random.randn(num_filters*Ha*Wa, hidden_dim)
        self.params['FCb1'] = np.zeros((hidden_dim,))
        for i in xrange(M):
            self.params['Bgamma'+str(2*i+1)] = weight_scale*np.random.randn(hidden_dim)
            self.params['Bbeta'+str(2*i+1)]  = np.zeros(hidden_dim)
            self.params['FCW'+str(i+2)] = weight_scale*np.random.randn(hidden_dim, hidden_dim)
            self.params['b4'+str(i+2)] = np.zeros((hidden_dim,))
            self.params['gamma'+str(2*i+2)] = weight_scale*np.random.randn(hidden_dim)
            self.params['beta'+str(2*i+2)]  = np.zeros(hidden_dim)

        # to out
        self.params['Wo'] = weight_scale*np.random.randn(hidden_dim, num_classes)
        self.params['bo'] = np.zeros((num_classes,))

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']

        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.parmas['gamma2'], self.params['beta2']
        gamma1, beta1 = self.params['gamma3'], self.params['beta3']
        gamma2, beta2 = self.parmas['gamma4'], self.params['beta4']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        bn_param = {'mode': 'train'}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        h, conv_cache1 = conv_forward_im2col(X, W1, b1, conv_param)
        h, sbtn_cache1 = spatial_batchnorm_forward(h, gamma1, beta1, bn_param)
        h, relu_cache1 = relu_forward(h)
        h, conv_cache2 = conv_forward_im2col(h, W2, b2, conv_param)
        b, sbtn_cache2 = spatial_batchnorm_forward(h, gamma2, beta2, bn_param)
        h, relu_cache2 = relu_forward(h)
        h, maxp_cache  = max_pool_forward_fast(h, pool_param)
        d1, d2, d3, d4 = h.shape
        h = h.reshape(h.shape[0], -1)
        h, affn_cache1 = affine_forward(h, W3, b3)
        b, batn_cache1 = batchnorm_forward(h, gamma3, beta3)
        h, relu_cache3 = relu_forward(h)
        h, affn_cache2 = affine_forward(h, W4, b4)
        h, batn_cache2 = batchnorm_forward(h, gamma4, beta4)
        h, relu_cache4 = relu_forward(h)
        h, affn_cache3 = affine_forward(h, W5, b5)
        scores = h

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dx = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.linalg.norm(self.params['W3'])**2 + \
                              np.linalg.norm(self.params['W4'])**2 + \
                              np.linalg.norm(self.params['W5'])**2)
        dx, dW5, db5 = affine_backward(dx, affn_cache3)
        dx = relu_backward(dx, relu_cache4)
        dx, dgamma4, dbeta4 = batchnorm_backward(dx, batn_cache2)
        dx, dW4, db4 = affine_backward(dx, affn_cache2)
        dx = relu_backward(dx, relu_cache3)
        dx, dgamma3, dbeta3 = batchnorm_backward(dx, batn_cache1)
        dx, dW3, db3 = affine_backward(dx, affn_cache1)
        dx = dx.reshape(d1, d2, d3, d4)
        dx = max_pool_backward_fast(dx, maxp_cache)
        dx = relu_backward(dx, relu_cache2)
        dx, dgamma2, dbeta2 = spatial_batchnorm_backward(dx, sbtn_cache2)
        dx, dW2, db2 = conv_backward_im2col(dx, conv_cache2)
        dx = relu_backward(dx, relu_cache1)
        dx, dgamma1, dbeta1 = spatial_batchnorm_backward(dx, sbtn_cache1)
        dx, dW1, db1 = conv_backward_im2col(dx, conv_cache1)

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W3'] = dW3 + self.reg*self.params['W3']
        grads['b3'] = db3
        grads['W4'] = dW4 + self.reg*self.params['W4']
        grads['b4'] = db4
        grads['W5'] = dW5 + self.reg*self.params['W5']
        grads['b5'] = db5

        grads['gamma1'] = dgamma1
        grads['beta1']  = dbeta1
        grads['gamma2'] = dgamma2
        grads['beta2']  = dbeta2
        grads['gamma3'] = dgamma3
        grads['beta3']  = dbeta3
        grads['gamma4'] = dgamma4
        grads['beta4']  = dbeta4

        return loss, grads
