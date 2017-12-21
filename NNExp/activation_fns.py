import numpy as np

class SigmoidActivation:
    def fn(self, input_sum):
        # clipping is done to avoid overflow warnings
        return 1.0 / (1.0 + np.exp(- np.clip(input_sum, -100, +100)))
        #return 1.0 / (1.0 + np.exp(- input_sum))

    def d_fn(self, input_sum):
        return self.fn(input_sum) * (1 - self.fn(input_sum))

class ReLUActivation:
    def fn(self, input_sum):
        return np.maximum(input_sum, 0)

    def d_fn(self, input_sum):
        return np.greater(input_sum, 0).astype(int)
