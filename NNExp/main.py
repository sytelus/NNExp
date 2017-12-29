import experiments
import numpy as np

np.seterr(all='raise')
np.random.seed(42)

#experiments.nielson_3layer_full_data()
#experiments.invgrad_3layer()
#experiments.neilson_5layer_200rows_100epochs()

#experiments.linear_data_classical()

#experiments.odd_even_data_classical()

experiments.chap2_full_data()
