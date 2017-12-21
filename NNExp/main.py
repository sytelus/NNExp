import mnist_dataset as mnist
import network as nn
import network_config
import numpy as np

np.seterr(all='raise')
np.random.seed(42)

labeled_data = mnist.MnistDataset.from_pickled_file()
config = network_config.NetworkConfig()
config.neuron_counts = [784, 30, 10]

net = nn.Network(config)
net.sgd(labeled_data)


# this network learns much slower than a shallow one.
'''
net = network2.Network([784, 30, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)
'''
