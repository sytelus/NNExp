import numpy as np
import network_config
import labeled_data
import network as nn

class TwinNetwork:
    def __init__(self, config : network_config.NetworkConfig):
        self.config = config
        self.networks = [nn.Network(config), nn.Network(config)]

    def train(self, train_data, validate_date):
        # intialize weights and biases
        for nw in self.networks:
            nw._init_nn_params()

        n = len(train_data)

        # run epochs
        for j in range(self.config.epochs):
            # for each epoch, form mini-batches
            np.random.shuffle(train_data)
            mini_batches = [
                train_data[k : k + self.config.batch_size]
                for k in range(0, n, self.config.batch_size)]

            nn_len = len(self.networks)

            nn_id = 0
            for batch in mini_batches:
                total_loss, db_sum, dw_sum = \
                    self.networks[nn_id % nn_len]._train_batch(batch, n)

                # make update to weights and biases for entire batch
                self.networks[(nn_id + 1) % nn_len]._update_nn_params(self.config, 
                    self.biases, self.weights, total_loss, db_sum, dw_sum, len(batch), n)

                nn_id += 1

            # output results
            print("Epoch %d : %d / %d" % (j, self.test(validate_date), len(validate_date)))

        # end of training
        print('done')

    def test(self, test_data):
        e = 0
        for nw in self.networks:
            e += nw.test(test_data)
        return e / len(self.networks)


