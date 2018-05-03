import numpy as np

class ClassicBackprop:
    def fn(self, config, biases, weights, x, y_true):
        b_bp = [np.zeros(b.shape) for b in biases]
        w_bp = [np.zeros(w.shape) for w in weights]

        # feed_forward
        layer_input = x
        layer_inputs = [x] # list to store all the layer_inputs, layer by layer
        input_sums = [] # list to store all the input_sum vectors, layer by layer
        for b, w in zip(biases, weights):
            input_sum = np.dot(w, layer_input) + b
            input_sums.append(input_sum)
            layer_input = config.activation_c.fn(input_sum)
            layer_inputs.append(layer_input)

        # backward pass

        '''
            delta_i = delta_i+1 * W * dActivation
            delta_L = dLoss * dActivation
            dW = -eta * delta_i * z
        '''

        loss = config.loss_c.fn(layer_inputs[-1], y_true)
        d_loss = config.loss_c.d_fn(layer_inputs[-1], y_true)
        d_activation = config.activation_c.d_fn(input_sums[-1])
        delta =  d_loss * d_activation # error signal
        delta_boost = config.delta_boost

        b_bp[-1] = delta
        w_bp[-1] = np.dot(delta, layer_inputs[-2].transpose())

        for l in range(2, len(config.neuron_counts)):
            input_sum = input_sums[-l]
            d_activation = config.activation_c.d_fn(input_sum)
            delta = np.dot(weights[-l+1].transpose(), delta) * d_activation * delta_boost
            b_bp[-l] = delta
            w_bp[-l] = np.dot(delta, layer_inputs[-l-1].transpose())

        return (b_bp, w_bp, loss)

