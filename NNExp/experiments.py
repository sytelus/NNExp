import mnist_dataset as mnist
import labeled_data as ld
import numpy as np
import network as nn
import network_config
import init_fns
import loss_fns
import activation_fns
import param_update_fns
import math

def chap1_full_data():
    labeled_data = mnist.MnistDataset.from_pickled_file()
    config = network_config.NetworkConfig()
    config.neuron_counts = [784, 30, 10]

    net = nn.Network(config)
    net.sgd(labeled_data)

def chap2_full_data():
    labeled_data = mnist.MnistDataset.from_pickled_file()
    config = network_config.NetworkConfig()
    config.init_c = init_fns.LeCunNormalInit()
    config.loss_c = loss_fns.QuadraticLoss()
    config.activation_c = activation_fns.SigmoidActivation()
    config.neuron_counts = [784, 30, 30, 30, 30, 10]
    config.eta = 0.1
    config.lmbda = 5

    net = nn.Network(config)
    net.sgd(labeled_data)

def chap1_full_data_alt_update():
    labeled_data = mnist.MnistDataset.from_pickled_file()
    config = network_config.NetworkConfig()
    config.param_update_c = param_update_fns.GdFixedParamUpdate()
    config.neuron_counts = [784, 30, 10]

    net = nn.Network(config)
    net.sgd(labeled_data)

def chap1_200rows_epochs100():
    labeled_data = mnist.MnistDataset.from_pickled_file(20)
    config = network_config.NetworkConfig()
    config.neuron_counts = [784, 30, 10]
    config.epochs = 100

    net = nn.Network(config)
    net.sgd(labeled_data)

def chap1_200rows_epochs50_3hidden():
    labeled_data = mnist.MnistDataset.from_pickled_file(20)
    config = network_config.NetworkConfig()
    config.init_c = init_fns.LeCunNormalInit()
    # config.activation_c = activation_fns.ReLUActivation()
    config.neuron_counts = [784, 30, 20, 15, 10]
    config.epochs = 50

    net = nn.Network(config)
    net.sgd(labeled_data)

def linear_data_classical():
    def get_data(count):
        inputs = [np.reshape(k, (1,1)) for k in np.random.uniform(-5, 5, count)]
        outputs = [[[0], [1]] if k[0] < 0 else [[1], [0]] for k in inputs]
        data = list(zip(inputs, outputs, range(len(inputs))))
        return data

    dataset = ld.LabeledData() 
    dataset.train = get_data(10)
    dataset.test = get_data(100)
    dataset.validate = get_data(100)

    config = network_config.NetworkConfig()
    #config.param_update_c = param_update_fns.GdFixedParamUpdate()
    config.neuron_counts = [1, 2]
    config.epochs = 20

    net = nn.Network(config)
    net.sgd(dataset)
    
    
def odd_even_data_classical():
    def get_data(count):
        inputs = [np.reshape(k, (1,1)) for k in np.random.uniform(-10, 10, count)]
        outputs = [[[0], [1]] if math.floor(math.fabs(k[0])) % 2 == 0 else [[1], [0]] for k in inputs]
        data = list(zip(inputs, outputs, range(len(inputs))))
        return data

    dataset = ld.LabeledData() 
    dataset.train = get_data(1000)
    dataset.test = get_data(100)
    dataset.validate = get_data(100)

    config = network_config.NetworkConfig()
    #config.param_update_c = param_update_fns.GdFixedParamUpdate()
    config.neuron_counts = [1, 10, 10, 2]
    config.epochs = 50

    net = nn.Network(config)
    net.sgd(dataset)