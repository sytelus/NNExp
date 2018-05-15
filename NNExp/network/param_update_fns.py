import numpy as np


class ClassicParamUpdate:
    def fn(self, config, biases, weights, total_loss,
           b_batch, w_batch, batch_len, train_len):
        regularization = 1 - config.eta * (config.lmbda / train_len)

        # this effectively averages the gradients
        eta_batch = config.eta / batch_len

        weights_new = [regularization * w - eta_batch * wb
                       for w, wb in zip(weights, w_batch)]
        biases_new = [b - eta_batch * bb
                      for b, bb in zip(biases, b_batch)]

        return (biases_new, weights_new)


class SquaredParamUpdate:
    def fn(self, config, biases, weights, total_loss,
           b_batch, w_batch, batch_len, train_len):
        regularization = 1 - config.eta * (config.lmbda / train_len)
        eta_batch = config.eta / batch_len

        weights_new = [regularization * w - eta_batch * wb * np.fabs(wb)
                       for w, wb in zip(weights, w_batch)]
        biases_new = [b - eta_batch * bb
                      for b, bb in zip(biases, b_batch)]

        return (biases_new, weights_new)


class InvGradParamUpdate:
    def fn(self, config, biases, weights, total_loss,
           b_batch, w_batch, batch_len, train_len):
        regularization = 1 - config.eta * (config.lmbda / train_len)
        eta_batch = 1  # config.eta / batch_len

        avg_loss = np.sum(total_loss) / batch_len
        weights_new = [
            np.clip(regularization * w + eta_batch * InvGradParamUpdate.inv_delta(avg_loss, wb, regularization * w),
                    -100, 100)
            for w, wb in zip(weights, w_batch)]
        biases_new = [np.clip(b + eta_batch * InvGradParamUpdate.inv_delta(avg_loss, bb, b), -100, 100)
                      for b, bb in zip(biases, b_batch)]

        return (biases_new, weights_new)

    @staticmethod
    def inv_delta(avg_loss, delta, max_val):
        # total_loss_abs = math.fabs(total_loss) / 3
        # max_val = np.fabs(max_val)
        # max_val = 0.1 * np.where(max_val < 1E-100, 1E-100, max_val)

        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            inv = np.nan_to_num(-avg_loss / delta)
            np.clip(inv, -0.01, 0.01, inv)
        return inv
