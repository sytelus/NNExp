import mnist_dataset as mnist
import network as nn
import network_config
import init_fns
import activation_fns
import param_update_fns

def chap1_full_data():
    labeled_data = mnist.MnistDataset.from_pickled_file()
    config = network_config.NetworkConfig()
    config.neuron_counts = [784, 30, 10]

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