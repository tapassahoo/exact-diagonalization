import sys
import numpy as np
from numpy cimport ndarray
cimport numpy as np
cimport cython

@cython.boundscheck(False) 

class Mv:
	
	def __init__(self, int JKeM, int size_grid, ndarray[np.float64_t, ndim=1, mode="c"] HrotKe not None, ndarray[np.float64_t, ndim=2, mode="c"] Vpot not None, ndarray[double complex, ndim=1, mode="c"] basisJKeM not None, int count):
		self.JKeM = JKeM
		self.size_grid = size_grid
		self.HrotKe = HrotKe
		self.Vpot = Vpot
		self.basisJKeM = basisJKeM
		self.count = count

	def __call__(self, ndarray[double complex, ndim=1, mode="c"] v not None):
		self.v=v

		cdef ndarray[complex, ndim=1, mode="c"] u = np.zeros((self.JKeM*self.JKeM), dtype=np.complex128)
  
		cdef int i1, i2, i1p, i2p, ig1, ig2;
		# oprate with K1
		for i2 in range(self.JKeM):
			for i1 in range(self.JKeM):
				for i1p in range(self.JKeM):
					#u[i2+i1*self.JKeM]+=np.complex(self.HrotKe[i1p+i1*self.JKeM],0.0)*self.v[i2+i1p*self.JKeM]
					u[i2+i1*self.JKeM]+=self.HrotKe[i1p+i1*self.JKeM]*self.v[i2+i1p*self.JKeM]

		# oprate with K2
		for i1 in range(self.JKeM):
			for i2 in range(self.JKeM):
				for i2p in range(self.JKeM):
					#u[i2+i1*self.JKeM]+=np.complex(self.HrotKe[i2p+i2*self.JKeM],0.0)*self.v[i2p+i1*self.JKeM]
					u[i2+i1*self.JKeM]+=self.HrotKe[i2p+i2*self.JKeM]*self.v[i2p+i1*self.JKeM]

		cdef ndarray[double complex, ndim=1, mode="c"] temp1 = np.zeros((self.JKeM*self.size_grid), dtype=np.complex128)
		cdef ndarray[double complex, ndim=1, mode="c"] temp2 = np.zeros((self.size_grid*self.size_grid), dtype=np.complex128)
		cdef ndarray[double complex, ndim=1, mode="c"] temp3 = np.zeros((self.JKeM*self.size_grid), dtype=np.complex128)

		for i1 in range(self.JKeM):
			for i2 in range(self.JKeM):
				for ig2 in range(self.size_grid):
					temp1[ig2+i1*self.size_grid]+=self.basisJKeM[ig2+i2*self.size_grid]*self.v[i2+i1*self.JKeM]

		for i1 in range(self.JKeM):
			for ig1 in range(self.size_grid):
				for ig2 in range(self.size_grid):
					temp2[ig2+ig1*self.size_grid]+=self.basisJKeM[ig1+i1*self.size_grid]*temp1[ig2+i1*self.size_grid]

		for ig1 in range(self.size_grid):
			for ig2 in range(self.size_grid):
				temp2[ig2+ig1*self.size_grid]=self.Vpot[ig1,ig2]*temp2[ig2+ig1*self.size_grid]

		for ig1 in range(self.size_grid):
			for i2 in range(self.JKeM):
				for ig2 in range(self.size_grid):
					temp3[ig1+i2*self.size_grid]+=temp2[ig2+ig1*self.size_grid]*np.conjugate(self.basisJKeM[ig2+i2*self.size_grid])

		cdef ndarray[double complex, ndim=1, mode="c"] vec = np.zeros((self.JKeM*self.JKeM), dtype=np.complex128)
		for i1 in range(self.JKeM):
			for i2 in range(self.JKeM):
				for ig1 in range(self.size_grid):
					vec[i2+i1*self.JKeM]+=temp3[ig1+i2*self.size_grid]*np.conjugate(self.basisJKeM[ig1+i1*self.size_grid])
		u=u+vec
		self.count = self.count+1
		print(self.count)
		sys.stdout.flush()
		return u
