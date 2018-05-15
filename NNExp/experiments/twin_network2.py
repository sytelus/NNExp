import network.network as nn
import numpy as np


class TwinNetwork:
    def __init__(self, config : nn.network_config.NetworkConfig):
        self.config = config
        self.networks = [nn.Network(config), nn.Network(config)]

    def train(self, train_data, validate_date):
        # intialize weights and biases
        self.networks[0]._init_nn_params()
        for nw in self.networks[1:]:
            nw._init_nn_params(self.networks[0])

        n = len(train_data)

        # run epochs
        for j in range(self.config.epochs):
            # for each epoch, form mini-batches
            np.random.shuffle(train_data)
            mini_batches = [
                train_data[k : k + self.config.batch_size]
                for k in range(0, n, self.config.batch_size)]

            nn_len = len(self.networks)
            
            # normal training loop
            for i, batch in enumerate(mini_batches):
                total_loss, db_sum, dw_sum = \
                    self.networks[i % nn_len]._train_batch(batch, n)
                # make update to weights and biases for entire batch
                self.networks[i % nn_len]._update_nn_params(
                    total_loss, db_sum, dw_sum, len(batch), n)

            # co-training loop
            for i, batch in enumerate(mini_batches):
                total_loss, db_sum, dw_sum = \
                    self.networks[i % nn_len]._train_batch(batch, n)
                # make update to weights and biases for entire batch
                self.networks[i % nn_len]._update_nn_params(
                    total_loss, db_sum, dw_sum, len(batch), n)

            # output results
            print("Epoch %d : %d / %d" % (j, self.test(validate_date), len(validate_date)))

        # end of training
        print('done')

    def _train_batch(self, batch, n):
        nn_len = len(self.networks)

        # divide batches in to sublist
        sub_batches = [batch[i::nn_len] for i in range(nn_len)]

        #each network trains on its own batch first
        for i in range(nn_len):
            total_loss, db_sum, dw_sum = self.networks[i]._train_batch(sub_batches[i], n)
            self.networks[i]._update_nn_params(total_loss, db_sum, dw_sum, len(sub_batches[i]), n)

        #each network now generates error signal on next network's batch 
        for i in range(nn_len):
            sub_batch = sub_batches[(i + 1) % nn_len]



            total_loss, db_sum, dw_sum = self.networks[i]._train_batch(sub_batches[i + 1], n)
            self.networks[i]._update_nn_params(total_loss, db_sum, dw_sum, len(sub_batches[i]), n)

    def _cotrain_batch(self, batch, n, net1, net2):
        # init sums with zeros
        dw_sum = [np.zeros(w.shape) for w in self.weights]
        db_sum = [np.zeros(b.shape) for b in self.biases]

        total_loss = 0
        # for data point in batch
        for x, y_true, id in batch:
            # do the backprop to find dW and dB
            db_batch, dw_batch, loss, *_ = self.config.backprop_c.fn(
                self.config, self.biases, self.weights, x, y_true)

            # accumulate result of the backprop on each batch
            total_loss += loss
            dw_sum = [wb + wbp for wb, wbp in zip(dw_sum, dw_batch)]
            db_sum = [bb + bbp for bb, bbp in zip(db_sum, db_batch)]

        return (total_loss, db_sum, dw_sum)

    def test(self, test_data):
        e = 0
        for nw in self.networks:
            e += nw.test(test_data)
        return e / len(self.networks)



