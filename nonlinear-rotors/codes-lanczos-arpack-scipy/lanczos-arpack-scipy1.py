#Final version of diagonalizing H2O-H2O
#
#  Features:
#   - compute the eigenvalues and wavefunctions for the full 6D problen
#	- consider only even K 
#
import sys
import math
import numpy as np
import pot
from scipy.sparse.linalg import eigsh, LinearOperator

def binom(n,k):
	"""
	calculate binomial coefficient
	"""
	minus = n-k
	if minus < 0:
		return 0
	else:
		return (np.math.factorial(n)/ (np.math.factorial(k) * np.math.factorial(minus)))

def off_diag (j,k):                        
	"""
	off diagonal <JKM|H|J'K'M'> #
	"""
	f = np.sqrt((j*(j+1)) - (k*(k+1)))     
	return f                               

def littleD(ldJ,ldmp,ldm,ldtheta):
	"""
	Compute d(m',m, theta) ie. little d-rotation matrix 
	"""
	teza =(np.math.factorial(ldJ+ldm)*np.math.factorial(ldJ-ldm)*np.math.factorial(ldJ+ldmp)*np.math.factorial(ldJ-ldmp))*1.0
	dval = np.sqrt(teza) 
	tempD = 0.

	#determine max v that will begin to give negative factorial arguments
	if ldJ - ldmp > ldJ + ldm:
		upper = ldJ-ldmp
	else:
		upper = ldJ+ldm

	#iterate over intergers that provide non-negative factorial arguments
	for v in range(upper+1):
		a = ldJ - ldmp - v
		b = ldJ + ldm - v
		c = v + ldmp - ldm
		if (a>=0) and (b>=0) and (c>=0):
			tempD = tempD + (((-1.0)**v)/(np.math.factorial(a)*np.math.factorial(b)*np.math.factorial(c)*np.math.factorial(v)))*((np.cos(ldtheta/2.))**(2.*ldJ+ldm-ldmp-2.*v))*((-np.sin(ldtheta/2.))**(ldmp-ldm+2.*v))
	return dval*tempD

class mv():
	def __init__(self, JKeM, size_grid, HrotKe, Vpot, basisJKeM, count):
		self.JKeM = JKeM
		self.size_grid = size_grid
		self.HrotKe = HrotKe
		self.Vpot = Vpot
		self.basisJKeM = basisJKeM
		self.count=count

	def __call__(self, v):
		self.v=v

		u=np.zeros((self.JKeM*self.JKeM), dtype=complex);
  
		# oprate with K1
		for i2 in range(self.JKeM):
			for i1 in range(self.JKeM):
				for i1p in range(self.JKeM):
					u[i2+i1*self.JKeM]+=self.HrotKe[i1p+i1*self.JKeM]*self.v[i2+i1p*self.JKeM]

		# oprate with K2
		for i1 in range(self.JKeM):
			for i2 in range(self.JKeM):
				for i2p in range(self.JKeM):
					u[i2+i1*self.JKeM]+=self.HrotKe[i2p+i2*self.JKeM]*self.v[i2p+i1*self.JKeM]

		# potential term
		temp1 = np.zeros((self.JKeM*self.size_grid),dtype=complex);
		temp2 = np.zeros((self.size_grid*self.size_grid),dtype=complex);
		temp3 = np.zeros((self.JKeM*self.size_grid),dtype=complex);

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

		vec=np.zeros((self.JKeM*self.JKeM), dtype=complex);
		for i1 in range(self.JKeM):
			for i2 in range(self.JKeM):
				for ig1 in range(self.size_grid):
					vec[i2+i1*self.JKeM]+=temp3[ig1+i2*self.size_grid]*np.conjugate(self.basisJKeM[ig1+i1*self.size_grid])
		u=u+vec
		self.count=self.count+1
		print(self.count)
		return u

if __name__ == '__main__':    
	write_log = True
	write_norm = False
	write_pot = False
	zCOM = sys.argv[1]
	Jmax=int(sys.argv[2])
	niter=int(sys.argv[3])

	size_theta = int(Jmax+1)
	size_phi = int(2*Jmax+3)
	size_grid = size_theta*size_phi*size_phi
	strFile = "arpack-2-p-H2O-jmax"+str(Jmax)+"-Rpt"+str(zCOM)+"Angstrom-grid"+str(size_theta)+"-"+str(size_phi)+"-maxiter"+str(niter)+".txt"
	path_dir = ""

	if (write_log):
		log_file = path_dir+"logout-"+strFile
		log_write = open(log_file,'w')
		log_write.write("Jmax = "+str(Jmax)+" \n")
		log_write.write("size_theta = "+str(size_theta)+" \n")
		log_write.write("size_phi = "+str(size_phi)+" \n")
	
	#print the normalization 
	tol = 10e-8

	#The rotational A, B, C constants are indicated by Ah2o, Bh2o and Ch2o, respectively. The unit is cm^-1. 
	Ah2o= 27.877 #cm-1 
	Bh2o= 14.512 #cm-1
	Ch2o= 9.285  #cm-1
	CMRECIP2KL = 1.4387672;       	# cm^-1 to Kelvin conversion factor
	Ah2o=Ah2o*CMRECIP2KL
	Bh2o=Bh2o*CMRECIP2KL
	Ch2o=Ch2o*CMRECIP2KL

	xGL,wGL = np.polynomial.legendre.leggauss(size_theta)              
	phixiGridPts = np.linspace(0,2*np.pi,size_phi, endpoint=False)  
	dphixi = 2.*np.pi/size_phi                                 
	if (write_log):
		log_write.write("#*********************************************************************\n")
		log_write.write("# A list of Gaussian quadrature points of Legendre polynomials - \n")
		log_write.write("\n")
		for i in range(size_theta):
			log_write.write(str(xGL[i])+"\n")
		log_write.write("\n")
		log_write.write("# A list of the corrsponding weights - \n")
		log_write.write("\n")
		for i in range(size_theta):
			log_write.write(str(wGL[i])+"\n")
		log_write.write("#*********************************************************************\n")
		log_write.write("# Gaussian quadrature points of phi and chi angles - \n")
		log_write.write("\n")
		for i in range(size_phi):
			log_write.write(str(phixiGridPts[i])+"\n")
		log_write.write("\n")
		log_write.write("# The corrsponding weights - \n")
		log_write.write("\n")
		log_write.write(str(dphixi)+"\n")
		log_write.write("#*********************************************************************\n")

	JKM = int(((2*Jmax+1)*(2*Jmax+2)*(2*Jmax+3)/6)) #JKM = "Sum[(2J+1)**2,{J,0,Jmax}]" is computed in mathematica

	if Jmax%2 ==0:
		JKeM = int((JKM+Jmax+1)/2)
		JKoM = int(JKM-JKeM)
	else:
		JKoM = int((JKM+Jmax+1)/2)
		JKeM = int(JKM-JKoM)

	if (write_log):
		log_write.write("# Number of |JKM> basis = "+str(JKM)+"\n")
		log_write.write("\n")
		log_write.write("# Number of even K in |JKM> = "+str(JKeM)+"\n")
		log_write.write("# Number of odd  K in |JKM> = "+str(JKoM)+"\n")
		log_write.write("#*********************************************************************\n")
    
	JKeMQuantumNumList = np.zeros((JKeM,3),int)

	#even
	jtempcounter = 0
	for J in range(Jmax+1):
		if J%2==0:
			for K in range(-J,J+1,2):
				for M in range(-J,J+1):
					JKeMQuantumNumList[jtempcounter,0]=J
					JKeMQuantumNumList[jtempcounter,1]=K
					JKeMQuantumNumList[jtempcounter,2]=M
					jtempcounter+=1
		else:
			for K in range(-J+1,J,2):
				for M in range(-J,J+1):
					JKeMQuantumNumList[jtempcounter,0]=J
					JKeMQuantumNumList[jtempcounter,1]=K
					JKeMQuantumNumList[jtempcounter,2]=M
					jtempcounter+=1

	#
	dJKeM = np.zeros((JKeM,size_theta),float)
	KJKeM = np.zeros((JKeM,size_phi),complex)
	MJKeM = np.zeros((JKeM,size_phi),complex)


	#block for construction of individual basis begins 
	Nk = 1.0
	for s in range(JKeM):
		for th in range(size_theta):
			dJKeM[s,th] = np.sqrt((2.*JKeMQuantumNumList[s,0]+1)/(8.*np.pi**2))*littleD(JKeMQuantumNumList[s,0],JKeMQuantumNumList[s,2],JKeMQuantumNumList[s,1],np.arccos(xGL[th]))*np.sqrt(wGL[th])

		for ph in range(size_phi):
			KJKeM[s,ph] = np.exp(1j*phixiGridPts[ph]*JKeMQuantumNumList[s,1])*np.sqrt(dphixi)*Nk
			MJKeM[s,ph] = np.exp(1j*phixiGridPts[ph]*JKeMQuantumNumList[s,2])*np.sqrt(dphixi)*Nk
	#block for construction of individual basis ends

	#block for construction of |J1K1M1,J2K2M2> basis begins 
	eEEbasisuse = KJKeM[:,np.newaxis,np.newaxis,:]*MJKeM[:,np.newaxis,:,np.newaxis]*dJKeM[:,:,np.newaxis,np.newaxis]
	eEEebasisuse = np.reshape(eEEbasisuse,(JKeM,size_grid),order='C')
	basisJKeM = np.reshape(eEEebasisuse,(JKeM*size_grid),order='C')
	#block for construction of |J1K1M1,J2K2M2> basis ends

	normMat = np.tensordot(eEEebasisuse, np.conjugate(eEEebasisuse), axes=([1],[1]))

	if (write_log):
		log_write.write("\n")
		log_write.write("# eEEbasisuse.shape: shape of the even <theta,phi,chi|JKM> basis: " + str(eEEbasisuse.shape)+" \n")
		log_write.write("# eEEebasisuse.shape: reduced shape of the even <size_grid|JKM> basis: " + str(eEEebasisuse.shape)+" \n")
		log_write.write("# normMat.shape: shape of the even <JKM|JKM> basis: " + str(normMat.shape)+" \n")
		log_write.write("#*********************************************************************\n")
		log_write.close()

	if (write_norm):
		norm_file = path_dir+"norm-check-"+strFile
		norm_write = open(norm_file,'w')
		for s1 in range(JKeM):
			for s2 in range(JKeM):
				if (np.abs(normMat[s1,s2]) > tol):
					norm_write.write("L vec Rotor1: "+str(JKeMQuantumNumList[s1,0])+" "+str(JKeMQuantumNumList[s1,1])+" "+str(JKeMQuantumNumList[s1,2])+"\n")
					norm_write.write("R vec Rotor1: "+str(JKeMQuantumNumList[s2,0])+" "+str(JKeMQuantumNumList[s2,1])+" "+str(JKeMQuantumNumList[s2,2])+"\n")
					norm_write.write("Norm: "+str(normMat[s1,s2])+"\n")
					norm_write.write("\n")
		norm_write.close()

	#Construction of potential matrix begins
	com1=[0.0,0.0,0.0]
	com2=[0.0,0.0,zCOM]
	v6d = np.zeros((size_grid,size_grid),dtype=float)
	for th1 in range(size_theta):
		for ph1 in range(size_phi):
			for ch1 in range(size_phi):
				ii = ch1+(ph1+th1*size_phi)*size_phi
				Eulang1=[phixiGridPts[ph1], math.acos(xGL[th1]), phixiGridPts[ch1]]

				for th2 in range(size_theta):
					for ph2 in range(size_phi):
						for ch2 in range(size_phi):
							jj = ch2+(ph2+th2*size_phi)*size_phi
							Eulang2=[phixiGridPts[ph2], math.acos(xGL[th2]), phixiGridPts[ch2]]
							v6d[ii,jj]=pot.caleng(com1,com2,Eulang1,Eulang2)
	#Construction of potential matrix ends

	# construction of kinetic energy matrix - BEGINS
	HrotKe = np.zeros((JKeM*JKeM),dtype=float)
    
	for jkm in range(JKeM):
		for jkmp in range(JKeM):
			jj = jkmp+jkm*JKeM
			if JKeMQuantumNumList[jkm,0]==JKeMQuantumNumList[jkmp,0] and JKeMQuantumNumList[jkm,2]==JKeMQuantumNumList[jkmp,2]:
				if JKeMQuantumNumList[jkm,1]==(JKeMQuantumNumList[jkmp,1]-2):
					HrotKe[jj] += 0.25*(Ah2o-Ch2o)*off_diag(JKeMQuantumNumList[jkm,0],JKeMQuantumNumList[jkm,1])*off_diag(JKeMQuantumNumList[jkm,0],JKeMQuantumNumList[jkm,1]+1)
				elif JKeMQuantumNumList[jkm,1]==(JKeMQuantumNumList[jkmp,1]+2):
					HrotKe[jj] += 0.25*(Ah2o-Ch2o)*off_diag(JKeMQuantumNumList[jkm,0],JKeMQuantumNumList[jkm,1]-1)*off_diag(JKeMQuantumNumList[jkm,0],JKeMQuantumNumList[jkm,1]-2)
				elif JKeMQuantumNumList[jkm,1]==(JKeMQuantumNumList[jkmp,1]):
					HrotKe[jj] += (0.5*(Ah2o + Ch2o)*(JKeMQuantumNumList[jkm,0]*(JKeMQuantumNumList[jkm,0]+1)) + (Bh2o - 0.5*(Ah2o+Ch2o)) * ((JKeMQuantumNumList[jkm,1])**2))
	# construction of kinetic energy matrix - ENDS
    
	count=0
	hv=mv(JKeM, size_grid, HrotKe, v6d, basisJKeM,count)
	A = LinearOperator((JKeM*JKeM,JKeM*JKeM), matvec=hv, dtype=complex)
	'''
	start_time = timeit.default_timer()
	for i in range(1):
		A.matvec(np.ones(JKeM*JKeM))
	print(timeit.default_timer() - start_time)

	exit()
	'''
	vals = eigsh(A, k=1, which='SA', maxiter=None, return_eigenvectors=False)
	energy_file = path_dir+"energy-levels-"+strFile
	energy_write = open(energy_file,'w')
	energy_write.write(str(vals)+"\n")
	energy_write.close()
