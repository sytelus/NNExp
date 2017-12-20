import numpy as np
import loss_fns
import init_fns
import activation_fns

class NetworkConfig:
    neuron_counts = []
    loss_c = loss_fns.QuadraticLoss()
    activation_c = activation_fns.SigmoidActivation()
    init_c = init_fns.NormalInit()
    epochs = 30
    batch_size = 10
    eta = 3 # learning rate
    lmbda = 0.0 # regularization
