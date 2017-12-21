import numpy as np
import loss_fns
import init_fns
import activation_fns
import backprop_fns
import param_update_fns

class NetworkConfig:
    neuron_counts = []
    loss_c = loss_fns.QuadraticLoss()
    activation_c = activation_fns.SigmoidActivation()
    init_c = init_fns.NormalInit()
    backprop_c = backprop_fns.ClassicBackprop()
    param_update_c = param_update_fns.ClassicParamUpdate()
    epochs = 30
    batch_size = 10
    eta = 3 # learning rate
    lmbda = 0.0 # regularization
