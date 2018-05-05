import numpy as np

from . import loss_fns
from . import init_fns
from . import activation_fns
from . import backprop_fns
from . import param_update_fns

class NetworkConfig:
    neuron_counts = []
    loss_c = loss_fns.QuadraticLoss()
    activation_c = activation_fns.SigmoidActivation()
    init_c = init_fns.NormalInit()
    backprop_c = backprop_fns.ClassicBackprop()
    param_update_c = param_update_fns.ClassicParamUpdate()
    epochs = 150
    delta_boost = 1
    batch_size = 10
    eta = 3 # learning rate
    lmbda = 0.0 # regularization
