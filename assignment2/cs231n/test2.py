import numpy as np
import numpy as np


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def forward(x, gamma, beta, h=0.0, idx=0):
    eps = 1e-5
    sample_mean = np.mean(x, axis=0)
    sample_mean[idx] += h
    sample_var  = np.var(x, axis=0)
    x_hat = (x - sample_mean)/np.sqrt(sample_var + eps)
    out = gamma*x_hat + beta
    cache = (x, x_hat, gamma, sample_mean, sample_var + eps)
    return out, cache


def backward(dx, cache):
  x = cache[0]
  x_hat = cache[1]
  gamma = cache[2]
  mu = cache[3]
  var = cache[4]
  N = x.shape[0]

  doutdx_hat = dout*gamma
  doutdvar = np.sum([doutdx_hat[i,:]*(x[i,:] - mu) for i in xrange(N)], axis=0)
  doutdvar *= -0.5*var**(-3/2)
  doutdmu = doutdvar*(-2/N)*np.sum(x - mu, axis=0) - (1/np.sqrt(var))*np.sum(doutdx_hat, axis=0)
  dx = doutdx_hat*(1/np.sqrt(var)) + doutdvar*(2/N)*(x - mu) + doutdmu*(1/N)
  dgamma = np.sum([dout[i,:]*x_hat[i,:] for i in xrange(N)], axis=0)
  dbeta = np.sum(dout, axis=0)

  return dx, dgamma, dbeta, doutdvar


N, D = 4, 5
x = 5 * np.random.randn(N, D) + 12
gamma = np.random.randn(D)
beta = np.random.randn(D)
dout = np.random.randn(N, D)
sigma = np.random.rand()

delta = 1e-5
l1, _ = forward(x, gamma, beta, h=delta)
l2, _ = forward(x, gamma, beta, h=-delta)
_, cache = forward(x, gamma, beta)
dx, dgamma, dbeta, doutdvar = backward(dout, cache)

dvar_num = (l1 - l2)/(2*delta)
print doutdvar
print dvar_num
print rel_error(dvar_num, doutdvar)
