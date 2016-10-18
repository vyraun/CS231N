from layers import *
from solver import Solver
from fast_layers import conv_forward_im2col
from classifiers.fc_net import FullyConnectedNet
from classifiers.cnn import ThreeLayerConvNet
from gradient_check import eval_numerical_gradient_array
from gradient_check import eval_numerical_gradient
from cs231n.data_utils import get_CIFAR10_data

import numpy as np


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data()

from cs231n.classifiers.covnet import CovNet

model = CovNet()
solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                optim_config={
                    'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()
