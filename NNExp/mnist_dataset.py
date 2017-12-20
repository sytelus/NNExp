import pickle
import numpy as np
import labeled_data as ld
import gzip
import sys
import os

class MnistDataset:
    @staticmethod
    def from_pickled_file(file_path = os.path.join('data', 'mnist.pkl.gz')):
        script_path = sys.path[0]
        full_file_path = os.path.join(script_path, file_path)

        with gzip.open(full_file_path, 'rb') as f:       
            training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

            dataset = ld.LabeledData()
            dataset.train = MnistDataset.from_pickle_data(training_data)
            dataset.test = MnistDataset.from_pickle_data(test_data)
            dataset.validate = MnistDataset.from_pickle_data(validation_data)

            return dataset

    @staticmethod
    def from_pickle_data(pickle_data):
        inputs = [np.reshape(x, (784, 1)) for x in pickle_data[0]]
        outputs = [MnistDataset.one_hot(y) for y in pickle_data[1]]
        return list(zip(inputs, outputs, range(len(inputs))))

    @staticmethod
    def one_hot(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e