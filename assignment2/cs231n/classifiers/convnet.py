import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top/(1 + z)


class ConvNet(object):
    def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=3,
                 hidden_dim=128, num_classes=10, weight_scale=1e-3, reg=0.0,
                 N=3, M=2, use_batchnorm=False, dtype=np.float32):
        self.params = {}
        self.extra_params = {}
        self.reg = reg
        self.dtype = dtype
        self.N = N
        self.M = M
        self.filter_size = filter_size
        self.use_batchnorm = use_batchnorm

        stride = 1
        pad = (filter_size - 1)/2
        C, H, W = input_dim
        for i in xrange(self.N):
            C = C if i == 0 else num_filters
            self.params['CW'+str(2*i+1)] = weight_scale*np.random.randn(num_filters,
                                            C, filter_size, filter_size)
            self.params['Cb'+str(2*i+1)] = weight_scale*np.ones((num_filters,))
            if self.use_batchnorm:
                self.params['Sgamma'+str(2*i+1)] = weight_scale*np.random.randn(num_filters)
                self.params['Sbeta'+str(2*i+1)]  = np.zeros((num_filters,))
                self.extra_params['Sbn_param'+str(2*i+1)] = {'mode': 'train'}

            self.params['CW'+str(2*i+2)] = weight_scale*np.random.randn(num_filters,
                                            num_filters, filter_size, filter_size)
            self.params['Cb'+str(2*i+2)] = weight_scale*np.ones((num_filters,))
            if self.use_batchnorm:
                self.params['Sgamma'+str(2*i+2)] = weight_scale*np.random.randn(num_filters)
                self.params['Sbeta'+str(2*i+2)]  = np.zeros((num_filters,))
                self.extra_params['Sbn_param'+str(2*i+2)] = {'mode': 'train'}
            # for max pooling
            H = (H - 2)/2 + 1
            W = (W - 2)/2 + 1

        self.params['FCW1'] = weight_scale*np.random.randn(num_filters*H*W, hidden_dim)
        self.params['FCb1'] = np.zeros((hidden_dim,))
        prev_hidden_dim = num_filters*H*W
        for i in xrange(self.M):
            self.params['FCW'+str(i+1)] = weight_scale*np.random.randn(prev_hidden_dim, hidden_dim)
            self.params['FCb'+str(i+1)] = weight_scale*np.ones((hidden_dim,))
            if self.use_batchnorm:
                self.params['Bgamma'+str(i+1)] = weight_scale*np.random.randn(hidden_dim)
                self.params['Bbeta'+str(i+1)]  = np.zeros((hidden_dim,))
                self.extra_params['Bbn_param'+str(i+1)] = {'mode': 'train'}
            prev_hidden_dim = hidden_dim

        # to out
        self.params['FCWo'] = weight_scale*np.random.randn(hidden_dim, num_classes)
        self.params['FCbo'] = np.zeros((num_classes,))

    def loss(self, X, y=None):
        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}
        bn_param = {'mode': 'train'}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        h = X
        cacheA = []
        for i in xrange(self.N):
            CWi1 = self.params['CW'+str(2*i+1)]
            Cbi1 = self.params['Cb'+str(2*i+1)]
            if self.use_batchnorm:
                gamma1 = self.params['Sgamma'+str(2*i+1)]
                beta1  = self.params['Sbeta'+str(2*i+1)]
                bn_param1 = self.extra_params['Sbn_param'+str(2*i+1)]
            CWi2 = self.params['CW'+str(2*i+2)]
            Cbi2 = self.params['Cb'+str(2*i+2)]
            if self.use_batchnorm:
                gamma2 = self.params['Sgamma'+str(2*i+2)]
                beta2  = self.params['Sbeta'+str(2*i+2)]
                bn_param2 = self.extra_params['Sbn_param'+str(2*i+2)]

            h, conv_cache1 = conv_forward_im2col(h, CWi1, Cbi1, conv_param)
            if self.use_batchnorm:
                h, stbn_cache1 = spatial_batchnorm_forward(h, gamma1, beta1, bn_param1)
            else:
                stbn_cache1 = None
            h, relu_cache1 = relu_forward(h)
            h, conv_cache2 = conv_forward_im2col(h, CWi2, Cbi2, conv_param)
            if self.use_batchnorm:
                h, stbn_cache2 = spatial_batchnorm_forward(h, gamma2, beta2, bn_param2)
            else:
                stbn_cache2 = None
            h, relu_cache2 = relu_forward(h)
            h, maxp_cache  = max_pool_forward_fast(h, pool_param)
            cacheA.append([conv_cache1, stbn_cache1, relu_cache1,
                          conv_cache2, stbn_cache2, relu_cache2, maxp_cache])

        d1, d2, d3, d4 = h.shape
        h = h.reshape(h.shape[0], -1)

        cacheB = []
        for i in xrange(self.M):
            FCWi = self.params['FCW'+str(i+1)]
            FCbi = self.params['FCb'+str(i+1)]
            if self.use_batchnorm:
                Bgamma = self.params['Bgamma'+str(i+1)]
                Bbeta  = self.params['Bbeta'+str(i+1)]
                bn_param = self.extra_params['Bbn_param'+str(i+1)]
            h, affn_cache = affine_forward(h, FCWi, FCbi)
            h, relu_cache = relu_forward(h)
            if self.use_batchnorm:
                h, batn_cache = batchnorm_forward(h, Bgamma, Bbeta, bn_param)
            else:
                batn_cache = None
            cacheB.append([affn_cache, relu_cache, batn_cache])

        FCWo = self.params['FCWo']
        FCbo = self.params['FCbo']
        h, affn_cache = affine_forward(h, FCWo, FCbo)
        scores = h

        if y is None:
            return scores

        loss, grads = 0, {}
        for k,v in self.params.iteritems():
            grads[k] = np.zeros_like(v)

        loss, dx = softmax_loss(scores, y)
        loss += 0.5*self.reg*np.sum([np.linalg.norm(param)**2
                                     for k, param in self.params.iteritems()
                                     if k.startswith('FCW')])
        dx, dFCWo, dFCbo = affine_backward(dx, affn_cache)
        grads['FCWo'] = dFCWo + self.reg*self.params['FCWo']
        grads['FCbo'] = dFCbo

        for i in reversed(xrange(self.M)):
            cache = cacheB[i]
            affn_cache = cache[0]
            relu_cache = cache[1]
            batn_cache = cache[2]
            if self.use_batchnorm:
                dx, dgamma, dbeta = batchnorm_backward(dx, batn_cache)
            dx = relu_backward(dx, relu_cache)
            dx, dFCWi, dFCbi = affine_backward(dx, affn_cache)

            grads['FCW'+str(i+1)] = dFCWi + self.reg*self.params['FCW'+str(i+1)]
            grads['FCb'+str(i+1)] = dFCbi
            if self.use_batchnorm:
                grads['Bgamma'+str(i+1)] = dgamma
                grads['Bbeta'+str(i+1)]  = dbeta

        dx = dx.reshape(d1, d2, d3, d4)
        for i in reversed(xrange(self.N)):
            cache = cacheA[i]
            conv_cache1 = cache[0]
            stbn_cache1 = cache[1]
            relu_cache1 = cache[2]
            conv_cache2 = cache[3]
            stbn_cache2 = cache[4]
            relu_cache2 = cache[5]
            maxp_cache  = cache[6]
            dx = max_pool_backward_fast(dx, maxp_cache)
            dx = relu_backward(dx, relu_cache2)
            if self.use_batchnorm:
                dx, dgamma2, dbeta2 = spatial_batchnorm_backward(dx, stbn_cache2)
            dx, dCWi2, dCbi2 = conv_backward_im2col(dx, conv_cache2)
            dx = relu_backward(dx, relu_cache1)
            if self.use_batchnorm:
                dx, dgamma1, dbeta1 = spatial_batchnorm_backward(dx, stbn_cache1)
            dx, dCWi1, dCbi1 = conv_backward_im2col(dx, conv_cache1)

            grads['CW'+str(2*i+1)] = dCWi1
            grads['Cb'+str(2*i+1)] = dCbi1
            if self.use_batchnorm:
                grads['Sgamma'+str(2*i+1)] = dgamma1
                grads['Sbeta'+str(2*i+1)]  = dbeta1
            grads['CW'+str(2*i+2)] = dCWi2
            grads['Cb'+str(2*i+2)] = dCbi2
            if self.use_batchnorm:
                grads['Sgamma'+str(2*i+2)] = dgamma2
                grads['Sbeta'+str(2*i+2)]  = dbeta2

        return loss, grads
