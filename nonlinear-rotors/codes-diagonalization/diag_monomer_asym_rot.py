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
import qpot
from scipy.sparse.linalg import eigs, eigsh
import cmath
import pkgdiag.diagfuns as dg

if __name__ == '__main__':    
	zCOM=float(sys.argv[1])
	Jmax=int(sys.argv[2])
	spin_isomer = sys.argv[3]

	size_theta = int(2*Jmax+3)
	size_phi = int(2*(2*Jmax+1))

	tol = 10e-8
	#print the normalization 
	norm_check = False
	io_write = False
	pot_write = False
	if (io_write == True):
		print("Jmax = ", Jmax, flush=True)
		print(" Number of theta grids = ", size_theta, flush=True)
		print(" Number of phi and chi grids = ", size_phi, flush=True)
		sys.stdout.flush()
	zCOM = '{:3.2f}'.format(zCOM)

	if (spin_isomer == "spinless"):
		isomer = "-" 
		basis_type = ""
	if (spin_isomer == "para"):
		isomer = "-p-" 
		basis_type = "even"
	if (spin_isomer == "ortho"):
		isomer = "-o-" 
		basis_type = "odd"

	strFile = "diag-2"+isomer+"H2O-one-rotor-fixed-cost-1-jmax"+str(Jmax)+"-Rpt"+str(zCOM)+"Angstrom-grids-"+str(size_theta)+"-"+str(size_phi)+"-saved-basis.txt"
	#prefile = "../exact-energies-of-H2O/"
	prefile = ""

	#The rotational A, B, C constants are indicated by Ah2o, Bh2o and Ch2o, respectively. The unit is cm^-1. 
	Ah2o= 27.877 #cm-1 
	Bh2o= 14.512 #cm-1
	Ch2o= 9.285  #cm-1
	CMRECIP2KL = 1.4387672;       	# cm^-1 to Kelvin conversion factor
	Ah2o=Ah2o*CMRECIP2KL
	Bh2o=Bh2o*CMRECIP2KL
	Ch2o=Ch2o*CMRECIP2KL

	xGL,wGL=np.polynomial.legendre.leggauss(size_theta)              
	phixiGridPts=np.linspace(0,2*np.pi,size_phi,endpoint=False)  
	dphixi=2.*np.pi/size_phi
	if (io_write == True):
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

	if ((Jmax%2) ==0):
		JKeM = int((JKM+Jmax+1)/2)
		JKoM = int(JKM-JKeM)
	else:
		JKoM = int((JKM+Jmax+1)/2)
		JKeM = int(JKM-JKoM)

	if (io_write == True):
		print("|------------------------------------------------")
		print("| Number of basis functions calculations ....")
		print("| ")
		print("| # of |JKM> basis = "+str(JKM))
		print("| ")
		print("| # of even K in |JKM> = "+str(JKeM))
		print("| # of odd  K in |JKM> = "+str(JKoM))
		print("| ")
		print("|------------------------------------------------")
		
	if (spin_isomer == "spinless"):
		njkm = JKM	
	if (spin_isomer == "para"):
		njkm = JKeM	
	if (spin_isomer == "ortho"):
		njkm = JKoM	

	njkmQuantumNumList = dg.get_numbbasis(njkm,Jmax,spin_isomer)

	dJKM, KJKM, MJKM = dg.wigner_basis(njkm,size_theta,size_phi,njkmQuantumNumList,xGL,wGL,phixiGridPts,dphixi)

	#block for construction of |J1K1M1,J2K2M2> basis begins 
	eEEbasisuse = KJKM[:,np.newaxis,np.newaxis,:]*MJKM[:,np.newaxis,:,np.newaxis]*dJKM[:,:,np.newaxis,np.newaxis]
	eEEebasisuse = np.reshape(eEEbasisuse,(njkm,size_theta*size_phi*size_phi),order='C')
	#block for construction of |J1K1M1,J2K2M2> basis ends

	if (norm_check == True):
		normMat = np.tensordot(eEEebasisuse, np.conjugate(eEEebasisuse), axes=([1],[1]))
		dg.normalization_check(prefile,strFile,basis_type,eEEbasisuse,eEEebasisuse,normMat,njkm,njkmQuantumNumList,tol)

	v1d = dg.get_pot(size_theta,size_phi,zCOM,xGL,phixiGridPts)

	tempa = v1d[np.newaxis,:]*eEEebasisuse
	Hpot = np.tensordot(np.conjugate(eEEebasisuse), tempa, axes=([1],[1]))

	if (pot_write == True):
		dg.normalization_check(prefile,strFile,basis_type,v1d,eEEebasisuse,Hpot,njkm,njkmQuantumNumList,tol)

	Hrot = dg.get_rotmat(njkm,njkmQuantumNumList,Ah2o,Bh2o,Ch2o)
    
	Htot = Hrot + Hpot
	if (np.all(np.abs(Htot-Htot.T) < tol) == False):
		print("The Hamiltonian matrx Htot is not hermitian.")
		exit()

	#Estimation of eigenvalues and eigenvectors begins here
	eigVal, eigVec = LA.eigh(Htot)
	sortIndex_eigVal = eigVal.argsort()     # prints out eigenvalues for pure asymmetric top rotor (z_ORTHOz)
	eigVal_sort = eigVal[sortIndex_eigVal]       
	eigVec_sort = eigVec[:,sortIndex_eigVal]       
	#Estimation of eigenvalues and eigenvectors ends here

	#printing block is opened
	eigVal_comb = np.array([eigVal_sort, eigVal_sort/CMRECIP2KL])

	eigVal_file = prefile+"eigen-values-"+strFile
	np.savetxt(eigVal_file, eigVal_comb.T, fmt='%20.8f', delimiter=' ', header='Energy levels of a aymmetric top - Units associated with the first and second columns are Kelvin and wavenumber, respectively. ')

	for idx in range(4):
		eigVecRe = np.real(np.dot(np.conjugate(eigVec_sort[:,idx].T),eigVec_sort[:,idx]))
		eigVecIm = np.imag(np.dot(np.conjugate(eigVec_sort[:,idx].T),eigVec_sort[:,idx]))
		print("Checking normalization of ground state eigenfunction - Re: "+str(eigVecRe)+" Im: "+str(eigVecIm))

		avgHpotL = np.dot(Hpot,eigVec_sort[:,idx])
		avgHpot = np.dot(np.conjugate(eigVec_sort[:,idx].T),avgHpotL)
		print("Expectation value of ground state potential - Re: "+str(avgHpot.real)+" Im: "+str(avgHpot.imag))
	# printing block is closed

	# printing block is opened
	idx=0
	avgHpotL = np.dot(Hpot,eigVec_sort[:,idx])
	avgHpot = np.dot(np.conjugate(eigVec_sort[:,idx].T),avgHpotL)

	gs_eng_file = prefile+"ground-state-energy-"+strFile
	gs_eng_write = open(gs_eng_file,'w')
	gs_eng_write.write("#Printing of ground state energies in inverse Kelvin - "+"\n")
	gs_eng_write.write('{0:1} {1:^19} {2:^20}'.format("#","<T+V>", "<V>"))
	gs_eng_write.write("\n")
	gs_eng_write.write('{0:^20.8f} {1:^20.8f}'.format(eigVal_sort[0], avgHpot.real))
	gs_eng_write.write("\n")
	gs_eng_write.close()
	# printing block is closed

	# computation of reduced density matrix
	reduced_density=np.zeros((njkm,Jmax+1),dtype=complex)
	for i in range(njkm):
		for ip in range(njkm):
			if ((njkmQuantumNumList[i,1]==njkmQuantumNumList[ip,1]) and (njkmQuantumNumList[i,2]==njkmQuantumNumList[ip,2])):
				reduced_density[i,njkmQuantumNumList[ip,0]]=np.conjugate(eigVec_sort[i,0])*eigVec_sort[ip,0]

	gs_ang_file = prefile+"ground-state-theta-distribution-"+strFile
	gs_ang_write = open(gs_ang_file,'w')
	gs_ang_write.write("#Printing of ground state theta distribution - "+"\n")
	gs_ang_write.write('{0:1} {1:^19} {2:^20}'.format("#","cos(theta)", "reduced density"))
	gs_ang_write.write("\n")
	
	sum3=complex(0.0,0.0)
	for th in range(size_theta):
		sum1=complex(0.0,0.0)
		for i in range(njkm):
			for ip in range(njkm):
				if ((njkmQuantumNumList[i,1]==njkmQuantumNumList[ip,1]) and (njkmQuantumNumList[i,2]==njkmQuantumNumList[ip,2])):
					sum1+=4.0*math.pi*math.pi*reduced_density[i,njkmQuantumNumList[ip,0]]*dJKM[i,th]*dJKM[ip,th]
		gs_ang_write.write('{0:^20.8f} {1:^20.8f}'.format(xGL[th],sum1.real/wGL[th]))
		gs_ang_write.write("\n")
		sum3+=sum1
	gs_ang_write.close()
	# printing block is closed
 	
	print("Normalization: reduced density matrix = ",sum3)
