import numpy as np


class ZerosInit:
    def get_biases(self, neuron_counts):
        return [np.zeros((y, 1)) for y in neuron_counts[1:]]

    def get_weights(self, neuron_counts):
        return [np.zeros((y, x))
                for x, y in zip(neuron_counts[:-1], neuron_counts[1:])]


class NormalInit:
    def get_biases(self, neuron_counts):
        return [np.random.randn(y, 1) for y in neuron_counts[1:]]

    def get_weights(self, neuron_counts):
        return [np.random.randn(y, x)
                for x, y in zip(neuron_counts[:-1], neuron_counts[1:])]


# https://keras.io/initializers/
class GlorotNormalInit:
    def get_biases(self, neuron_counts):
        return [np.random.randn(y, 1) for y in neuron_counts[1:]]

    def get_weights(self, neuron_counts):
        return [np.random.randn(y, x) / np.sqrt(2 / (x + y))
                for x, y in zip(neuron_counts[:-1], neuron_counts[1:])]


class LeCunNormalInit:
    def get_biases(self, neuron_counts):
        return [np.random.randn(y, 1) for y in neuron_counts[1:]]

    def get_weights(self, neuron_counts):
        return [np.random.randn(y, x) / np.sqrt(x)
                for x, y in zip(neuron_counts[:-1], neuron_counts[1:])]


class HeNormalInit:
    def get_biases(self, neuron_counts):
        return [np.random.randn(y, 1) for y in neuron_counts[1:]]

    def get_weights(self, neuron_counts):
        return [np.random.randn(y, x) / np.sqrt(2 / x)
                for x, y in zip(neuron_counts[:-1], neuron_counts[1:])]
