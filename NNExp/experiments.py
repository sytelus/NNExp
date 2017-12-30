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

def nielson_3layer_full_data():
    labeled_data = mnist.MnistDataset.from_pickled_file()
    config = network_config.NetworkConfig()
    config.neuron_counts = [784, 30, 10]

    net = nn.Network(config)
    net.sgd(labeled_data.train, labeled_data.validate)

def nielson_3layer_300rows():
    labeled_data = mnist.MnistDataset.from_pickled_file(30)
    config = network_config.NetworkConfig()
    config.neuron_counts = [784, 30, 10]

    net = nn.Network(config)
    net.sgd(labeled_data.train, labeled_data.validate)

def sq_grad_3layer_300rows():
    labeled_data = mnist.MnistDataset.from_pickled_file(30, True)
    config = network_config.NetworkConfig()
    config.neuron_counts = [784, 30, 10]
    config.param_update_c = param_update_fns.SquaredParamUpdate()
    config.eta = 18
    config.epochs = 100

    net = nn.Network(config)
    net.sgd(labeled_data.train, labeled_data.test)

def nielson_6layer_full_data():
    labeled_data = mnist.MnistDataset.from_pickled_file(30)
    config = network_config.NetworkConfig()
    config.init_c = init_fns.LeCunNormalInit()
    #config.loss_c = loss_fns.QuadraticLoss()
    #config.activation_c = activation_fns.SigmoidActivation()
    config.neuron_counts = [784, 30, 30, 30, 30, 10]
    config.eta = 0.001
    config.lmbda = 0.025
    config.epochs = 500

    net = nn.Network(config)
    net.sgd(labeled_data.train, labeled_data.validate)

def invgrad_6layer():
    labeled_data = mnist.MnistDataset.from_pickled_file(30)
    config = network_config.NetworkConfig()
    config.param_update_c = param_update_fns.InvGradParamUpdate()
    config.neuron_counts = [784, 30, 30, 30, 30, 10]
    config.lmbda = 0.025
    config.epochs = 500

    net = nn.Network(config)
    net.sgd(labeled_data.train, labeled_data.validate)

def invgrad_3layer():
    labeled_data = mnist.MnistDataset.from_pickled_file()
    config = network_config.NetworkConfig()
    config.param_update_c = param_update_fns.InvGradParamUpdate()
    config.neuron_counts = [784, 30, 10]

    net = nn.Network(config)
    net.sgd(labeled_data.train, labeled_data.validate)

def neilson_3layer_200rows_100epochs():
    labeled_data = mnist.MnistDataset.from_pickled_file(20)
    config = network_config.NetworkConfig()
    config.neuron_counts = [784, 30, 10]
    config.epochs = 100

    net = nn.Network(config)
    net.sgd(labeled_data.train, labeled_data.validate)

def neilson_5layer_200rows_100epochs():
    labeled_data = mnist.MnistDataset.from_pickled_file(20)
    config = network_config.NetworkConfig()
    config.init_c = init_fns.LeCunNormalInit()
    # config.activation_c = activation_fns.ReLUActivation()
    config.neuron_counts = [784, 30, 20, 15, 10]
    config.epochs = 50

    net = nn.Network(config)
    net.sgd(labeled_data.train, labeled_data.validate)

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
    #config.param_update_c = param_update_fns.InvGradParamUpdate()
    config.neuron_counts = [1, 3, 5, 2]
    config.epochs = 50

    net = nn.Network(config)
    net.sgd(dataset)
    
    
def odd_even_data_classical():
    def get_data(count):
        inputs = [np.reshape(k, (1,1)) for k in np.random.uniform(-10, 10, count)]
        outputs = [[[0], [1]] if math.floor(math.fabs(k[0])) % 2 == 0 else [[1], [0]] for k in inputs]
        data = list(zip(inputs, outputs, range(len(inputs))))
        return data

    dataset = ld.LabeledData() 
    dataset.train = get_data(10000)
    dataset.test = get_data(1000)
    dataset.validate = get_data(1000)

    config = network_config.NetworkConfig()
    #config.param_update_c = param_update_fns.InvGradParamUpdate()
    config.neuron_counts = [1, 30, 15, 5, 2]
    config.epochs = 50

    net = nn.Network(config)
    net.sgd(dataset)