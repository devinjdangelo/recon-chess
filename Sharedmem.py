from mpi4py import MPI 
import numpy as np 
from operator import mul
from functools import reduce

comm = MPI.COMM_WORLD 
rank = comm.rank
workers  = comm.Get_size()

def SharedArray(shape,dtype=np.float32):
		bytesize = np.dtype(dtype).itemsize
		try:
			nbytes = reduce(mul,shape,1)*bytesize
		except TypeError:
			nbytes = shape*bytesize
		#nbytes /= workers
		if rank==0:
			win = MPI.Win.Allocate_shared(nbytes, bytesize, comm=comm) 
		else:
			win = MPI.Win.Allocate_shared(0, bytesize, comm=comm) 
		buf, itemsize = win.Shared_query(0) 
		return np.ndarray(buffer=buf, dtype=np.float32, shape=shape) 

