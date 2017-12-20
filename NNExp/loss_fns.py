import numpy as np

class QuadraticLoss:
    def fn(self, y_pred, y_true):
        return 0.5*np.linalg.norm(y_pred-y_true)**2

    def d_fn(self, y_pred, y_true):
        return (y_pred-y_true)

class CrossEntropyLoss:
    def fn(self, y_pred, y_true):
        """Return the cost associated with an output ``y_pred`` and desired output
        ``y_true``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``y_pred`` and ``y_true`` have y_pred 1.0
        in the same slot, then the expression (1-y_true)*np.log(1-y_pred)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred)))

    def d_fn(self, y_pred, y_true):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        denom = y_pred * (y_pred-1)
        # denom = [d if d != 0 else [1E-30] for d in denom]
        return (y_pred-y_true) / denom

