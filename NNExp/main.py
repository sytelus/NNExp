import experiments
import numpy as np

np.seterr(all='raise')
np.seterr(under='ignore')

np.random.seed(42)

experiments.chap1_full_data_invgrad()
#experiments.chap1_full_data_alt_update()
#experiments.chap1_200rows_epochs50_3hidden()

#experiments.linear_data_classical()

#experiments.odd_even_data_classical()

#experiments.chap2_full_data()
