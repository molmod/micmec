import numpy as np
import h5py

h5_fn = "testing.h5"

with h5py.File(h5_fn, mode = 'a') as h5_f:
    
    atomic_number = 6 # carbon
    num_nodes = len(np.array(h5_f['system/pos']))
    h5_f['system/numbers'] = atomic_number*np.ones((num_nodes,))
