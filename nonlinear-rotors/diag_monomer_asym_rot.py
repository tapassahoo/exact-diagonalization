#Final version of diagonalizing H2O-H2O
#
#  Features:
#   - compute the eigenvalues and wavefunctions for the full 6D problen
#	- consider only even K 
#
import sys
import math
import numpy as np
from scipy import linalg as LA
import pot
from scipy.sparse.linalg import eigs, eigsh

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

if __name__ == '__main__':    
	strength=float(sys.argv[1])
	Jmax=int(sys.argv[2])
	angleNum = int(sys.argv[3])
	print("Jmax = ", Jmax, flush=True)
	print("angleNum = ", angleNum, flush=True)
	strFile = "-Field-Strength-"+str(strength)+"Kinv-N1-Jmax-"+str(Jmax)+"-grid-"+str(angleNum)+".txt"
	
	#print the normalization 
	write_pot = False
	tol = 10e-8

	#The rotational A, B, C constants are indicated by Ah2o, Bh2o and Ch2o, respectively. The unit is cm^-1. 
	Ah2o= 27.877 #cm-1 
	Bh2o= 14.512 #cm-1
	Ch2o= 9.285  #cm-1
	CMRECIP2KL = 1.4387672;       	# cm^-1 to Kelvin conversion factor
	Ah2o=Ah2o*CMRECIP2KL
	Bh2o=Bh2o*CMRECIP2KL
	Ch2o=Ch2o*CMRECIP2KL

	thetaNum = int(angleNum+1)                                           
	xGL,wGL = np.polynomial.legendre.leggauss(thetaNum)              
	phixiGridPts = np.linspace(0,2*np.pi,angleNum, endpoint=False)  
	dphixi = 2.*np.pi / angleNum                                 
	print("|------------------------------------------------")
	print("| A list of Gaussian quadrature points of Legendre polynomials - ")
	print("")
	print(xGL)
	print("")
	print("| A list of the corrsponding weights - ")
	print("")
	print(wGL)
	print("|------------------------------------------------")
	sys.stdout.flush()

	JKM = int(((2*Jmax+1)*(2*Jmax+2)*(2*Jmax+3)/6)) #JKM = "Sum[(2J+1)**2,{J,0,Jmax}]" is computed in mathematica

	if Jmax%2 ==0:
		JKeM = int((JKM+Jmax+1)/2)
		JKoM = int(JKM-JKeM)
	else:
		JKoM = int((JKM+Jmax+1)/2)
		JKeM = int(JKM-JKoM)

	print("|------------------------------------------------")
	print("| Number of basis functions calculations ....")
	print("| ")
	print("| # of |JKM> basis = "+str(JKM))
	print("| ")
	print("| # of even K in |JKM> = "+str(JKeM))
	print("| # of odd  K in |JKM> = "+str(JKoM))
	print("| ")
	print("|------------------------------------------------")
    
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

	# Compute littleD(j,m,k,theta) and compare it with the date estimated by asymrho.f
	"""
	theta = 1.0 # in degree
	for s in range(JKeM):
		print("j=",JKeMQuantumNumList[s,0],"m=",JKeMQuantumNumList[s,2],"k=",JKeMQuantumNumList[s,1],littleD(JKeMQuantumNumList[s,0],JKeMQuantumNumList[s,2],JKeMQuantumNumList[s,1],theta*np.pi/180.))
	"""

	#
	dJKeM = np.zeros((JKeM,len(xGL)),float)
	KJKeM = np.zeros((JKeM,angleNum),complex)
	MJKeM = np.zeros((JKeM,angleNum),complex)


	#block for construction of individual basis begins 
	Nk = 1.0
	for s in range(JKeM):
		for th in range(len(xGL)):
			dJKeM[s,th] = np.sqrt((2.*JKeMQuantumNumList[s,0]+1)/(8.*np.pi**2))*littleD(JKeMQuantumNumList[s,0],JKeMQuantumNumList[s,2],JKeMQuantumNumList[s,1],np.arccos(xGL[th]))*np.sqrt(wGL[th])

		for ph in range(angleNum):
			KJKeM[s,ph] = np.exp(1j*phixiGridPts[ph]*JKeMQuantumNumList[s,1])*np.sqrt(dphixi)*Nk
			MJKeM[s,ph] = np.exp(1j*phixiGridPts[ph]*JKeMQuantumNumList[s,2])*np.sqrt(dphixi)*Nk
	#block for construction of individual basis ends

	#block for construction of |J1K1M1,J2K2M2> basis begins 
	eEEbasisuse = KJKeM[:,np.newaxis,np.newaxis,:]*MJKeM[:,np.newaxis,:,np.newaxis]*dJKeM[:,:,np.newaxis,np.newaxis]
	eEEebasisuse = np.reshape(eEEbasisuse,(JKeM,len(xGL)*angleNum*angleNum),order='C')
	#block for construction of |J1K1M1,J2K2M2> basis ends

	#printing block is opened
	normMat = np.tensordot(eEEebasisuse, np.conjugate(eEEebasisuse), axes=([1],[1]))
	norm_check_file = "norm-check"+strFile
	norm_check_write = open("Exact-energies-of-H2O/"+norm_check_file,'w')
	norm_check_write.write("eEEbasisuse.shape: shape of the even |JKM> basis: " + str(eEEbasisuse.shape)+" \n")
	norm_check_write.write("eEEebasisuse.shape: reduced shape of the even |JKM> basis: " + str(eEEebasisuse.shape)+" \n")
	norm_check_write.write("normMat.shape: shape of the even <JKM|JKM> basis: " + str(normMat.shape)+" \n")
	norm_check_write.write("\n")
	norm_check_write.write("\n")

	for s1 in range(JKeM):
		for s2 in range(JKeM):
			if (np.abs(normMat[s1,s2]) > tol):
				norm_check_write.write("L vec Rotor1: "+str(JKeMQuantumNumList[s1,0])+" "+str(JKeMQuantumNumList[s1,1])+" "+str(JKeMQuantumNumList[s1,2])+"\n")
				norm_check_write.write("R vec Rotor1: "+str(JKeMQuantumNumList[s2,0])+" "+str(JKeMQuantumNumList[s2,1])+" "+str(JKeMQuantumNumList[s2,2])+"\n")
				norm_check_write.write("Norm: "+str(normMat[s1,s2])+"\n")
				norm_check_write.write("\n")
	norm_check_write.close()
	#printing block is closed


	#Computation of rotational energy of a asymmetric top molecule
	#Construction of potential matrix begins
	v1d = np.zeros(len(xGL)*angleNum*angleNum,float)
	ii = 0
	for th1 in range(len(xGL)):
		for ph1 in range(angleNum):
			for ch1 in range(angleNum):
				v1d[ii]=-strength*xGL[th1]
				ii = ii + 1
	#Construction of potential matrix ends

	#Construction of a constant potential matrix over the three Euler angles
	tempa = v1d[np.newaxis,:]*eEEebasisuse
	HpotKe = np.tensordot(np.conjugate(eEEebasisuse), tempa, axes=([1],[1]))

	#printing block is opened
	pot_check_file = "pot-check"+strFile
	pot_check_write = open("Exact-energies-of-H2O/"+pot_check_file,'w')
	pot_check_write.write("Printing of shapes and elements of potential matrix - "+"\n")
	pot_check_write.write("\n")
	pot_check_write.write("\n")
	pot_check_write.write("shape of potential matrix over three Euler angles : " + str(v1d.shape)+" \n")
	pot_check_write.write("eEEebasisuse.shape: reduced shape of the even |JKM> basis: " + str(eEEebasisuse.shape)+" \n")
	pot_check_write.write("shape of HpotKe : " + str(HpotKe.shape)+" \n")
	pot_check_write.write("\n")
	pot_check_write.write("\n")

	if (write_pot == True):
		for s1 in range(JKeM):
			for s2 in range(JKeM):
				if (np.abs(HpotKe[s1,s2]) > tol):
					pot_check_write.write("L vec Rotor1: "+str(JKeMQuantumNumList[s1,0])+" "+str(JKeMQuantumNumList[s1,1])+" "+str(JKeMQuantumNumList[s1,2])+"\n")
					pot_check_write.write("R vec Rotor1: "+str(JKeMQuantumNumList[s2,0])+" "+str(JKeMQuantumNumList[s2,1])+" "+str(JKeMQuantumNumList[s2,2])+"\n")
					pot_check_write.write("Constant potential field - Re: "+str(np.real(HpotKe[s1,s2]))+"   Im: "+str(np.imag(HpotKe[s1,s2]))+"\n")
					pot_check_write.write("\n")
	pot_check_write.close()
	# printing block is closed


	# construction of kinetic energy matrix - BEGINS
	HrotKe = np.zeros((JKeM,JKeM),dtype=float)
    
	for jkm in range(JKeM):
		for jkmp in range(JKeM):
			if JKeMQuantumNumList[jkm,0]==JKeMQuantumNumList[jkmp,0] and JKeMQuantumNumList[jkm,2]==JKeMQuantumNumList[jkmp,2]:
				if JKeMQuantumNumList[jkm,1]==(JKeMQuantumNumList[jkmp,1]-2):
					HrotKe[jkm,jkmp] += 0.25*(Ah2o-Ch2o)*off_diag(JKeMQuantumNumList[jkm,0],JKeMQuantumNumList[jkm,1])*off_diag(JKeMQuantumNumList[jkm,0],JKeMQuantumNumList[jkm,1]+1)
				elif JKeMQuantumNumList[jkm,1]==(JKeMQuantumNumList[jkmp,1]+2):
					HrotKe[jkm,jkmp] += 0.25*(Ah2o-Ch2o)*off_diag(JKeMQuantumNumList[jkm,0],JKeMQuantumNumList[jkm,1]-1)*off_diag(JKeMQuantumNumList[jkm,0],JKeMQuantumNumList[jkm,1]-2)
				elif JKeMQuantumNumList[jkm,1]==(JKeMQuantumNumList[jkmp,1]):
					HrotKe[jkm,jkmp] += (0.5*(Ah2o + Ch2o)*(JKeMQuantumNumList[jkm,0]*(JKeMQuantumNumList[jkm,0]+1)) + (Bh2o - 0.5*(Ah2o+Ch2o)) * ((JKeMQuantumNumList[jkm,1])**2))
	# construction of kinetic energy matrix - ENDS
    
	HtotKe = HrotKe + HpotKe
	if (np.all(np.abs(HtotKe-HtotKe.T) < tol) == False):
		print("The Hamiltonian matrx HtotKe is not hermitian.")
		exit()

	#Estimation of eigenvalues and eigenvectors begins here
	eigValKe, eigVecKe = LA.eigh(HtotKe)
	sortIndex_eigValKe = eigValKe.argsort()     # prints out eigenvalues for pure asymmetric top rotor (z_ORTHOz)
	eigValKe_sort = eigValKe[sortIndex_eigValKe]       
	eigVecKe_sort = eigVecKe[:,sortIndex_eigValKe]       
	#Estimation of eigenvalues and eigenvectors ends here

	#printing block is opened
	eigValKe_comb = np.array([eigValKe_sort, eigValKe_sort/CMRECIP2KL])

	eigValKe_file = "eigen-values"+strFile
	np.savetxt("Exact-energies-of-H2O/"+eigValKe_file, eigValKe_comb.T, fmt='%20.8f', delimiter=' ', header='Energy levels of a aymmetric top - Units associated with the first and second columns are Kelvin and wavenumber, respectively. ')

	for idx in range(4):
		eigVecKeRe = np.real(np.dot(np.conjugate(eigVecKe_sort[:,idx].T),eigVecKe_sort[:,idx]))
		eigVecKeIm = np.imag(np.dot(np.conjugate(eigVecKe_sort[:,idx].T),eigVecKe_sort[:,idx]))
		print("Checking normalization of ground state eigenfunction - Re: "+str(eigVecKeRe)+" Im: "+str(eigVecKeIm))

		avgHpotKeL = np.dot(HpotKe,eigVecKe_sort[:,idx])
		avgHpotKe = np.dot(np.conjugate(eigVecKe_sort[:,idx].T),avgHpotKeL)
		print("Expectation value of ground state potential - Re: "+str(avgHpotKe.real)+" Im: "+str(avgHpotKe.imag))
	# printing block is closed

	# printing block is opened
	idx=0
	avgHpotKeL = np.dot(HpotKe,eigVecKe_sort[:,idx])
	avgHpotKe = np.dot(np.conjugate(eigVecKe_sort[:,idx].T),avgHpotKeL)

	gs_eng_file = "ground-state-energies"+strFile
	gs_eng_write = open("Exact-energies-of-H2O/"+gs_eng_file,'w')
	gs_eng_write.write("#Printing of ground state energies in inverse Kelvin - "+"\n")
	gs_eng_write.write('{0:1} {1:^19} {2:^20}'.format("#","<T+V>", "<V>"))
	gs_eng_write.write("\n")
	gs_eng_write.write('{0:^20.8f} {1:^20.8f}'.format(eigValKe_sort[0], avgHpotKe.real))
	gs_eng_write.close()
	# printing block is closed

