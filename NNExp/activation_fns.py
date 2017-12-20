import numpy as np

class SigmoidActivation:
    def fn(self, input_sum):
        # clipping is done to avoid overflow warnings
        return 1.0 / (1.0 + np.exp(- np.clip(input_sum, -100, +100)))

    def d_fn(self, input_sum):
        return self.fn(input_sum * (1 - self.fn(input_sum)))
