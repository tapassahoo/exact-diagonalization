#Final version of diagonalizing H2O-H2O
#
#  Features:
#   - compute the eigenvalues and wavefunctions for the full 6D problen
#	- consider only even K 
#
import sys
import math
import numpy as np
import scipy
from scipy import linalg as LA
import qpot
from scipy.sparse.linalg import eigs, eigsh
import cmath
import pkgdiag_linear.diaglinear as dg

if __name__ == '__main__':    
	strength=float(sys.argv[1])
	Jmax=int(sys.argv[2])
	spin_isomer = sys.argv[3]

	size_theta = int(2*Jmax+5)
	size_phi = int(2*(2*Jmax+5))

	tol = 10e-8
	#print the normalization 
	io_write = False
	norm_check = False
	pot_write = False
	if (io_write == True):
		print("Jmax = ", Jmax, flush=True)
		print(" Number of theta grids = ", size_theta, flush=True)
		print(" Number of phi and chi grids = ", size_phi, flush=True)
		sys.stdout.flush()

	if (spin_isomer == "spinless"):
		isomer = "-" 
		basis_type = ""
	if (spin_isomer == "para"):
		isomer = "-p-" 
		basis_type = "even"
	if (spin_isomer == "ortho"):
		isomer = "-o-" 
		basis_type = "odd"

	strFile = "of-1"+isomer+"H2-jmax"+str(Jmax)+"-Field-Strength"+str(strength)+"Kelvin-grids-"+str(size_theta)+"-"+str(size_phi)+"-diag.txt"
	#prefile = "../exact-energies-of-H2O/"
	prefile = ""

	Bconst = 60.853 #cm-1 Taken from NIST data https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=1000
	CMRECIP2KL = 1.4387672;       	# cm^-1 to Kelvin conversion factor
	Bconst = Bconst*CMRECIP2KL
	
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

	JM = int((Jmax+1)**2) #JKM = "Sum[(2J+1)**2,{J,0,Jmax}]" is computed in mathematica

	if ((Jmax%2) ==0):
		JeM = int((JM+Jmax+1)/2)
		JoM = int(JM-JeM)
	else:
		JoM = int((JM+Jmax+1)/2)
		JeM = int(JM-JoM)

	if (io_write == True):
		print("|------------------------------------------------")
		print("| Number of basis functions calculations ....")
		print("| ")
		print("| # of |JM> basis = "+str(JM))
		print("| ")
		print("| # of even J in |JM> = "+str(JeM))
		print("| # of odd  J in |JM> = "+str(JoM))
		print("| ")
		print("|------------------------------------------------")
		
	if (spin_isomer == "spinless"):
		njm = JM	
	if (spin_isomer == "para"):
		njm = JeM	
	if (spin_isomer == "ortho"):
		njm = JoM	

	njmQuantumNumList = dg.get_numbbasisLinear(njm,Jmax,spin_isomer)

	basisfun=dg.spherical_harmonicsReal(njm,size_theta,size_phi,njmQuantumNumList,xGL,wGL,phixiGridPts,dphixi)

	if (norm_check == True):
		normMat = np.tensordot(basisfun, np.conjugate(basisfun), axes=([0],[0]))
		dg.normalization_checkLinear(prefile,strFile,basis_type,basisfun,normMat,njm,njmQuantumNumList,tol)

	basisfun1=dg.spherical_harmonicsComp(njm,size_theta,size_phi,njmQuantumNumList,xGL,wGL,phixiGridPts,dphixi)
	normMat1 = np.tensordot(basisfun, basisfun1, axes=([0],[0]))

	Hrot1=np.zeros((njm,njm),float)
	for jm in range(njm):
		for jmp in range(njm):
			sum=0.0
			for s in range(njm):
				sum+=np.real(normMat1[jm,s]*np.conjugate(normMat1[jmp,s]))*Bconst*njmQuantumNumList[s,0]*(njmQuantumNumList[s,0]+1.0)
			Hrot1[jm,jmp]=sum


	v1d=np.zeros((size_theta*size_phi),float)
	for th in range(size_theta):
		for ph in range(size_phi):
			v1d[ph+th*size_phi] = -strength*xGL[th] #A*cos(theta)

	tempa = v1d[:,np.newaxis]*basisfun
	Hpot = np.tensordot(np.conjugate(basisfun), tempa, axes=([0],[0]))

	if (pot_write == True):
		dg.normalization_checkLinear(prefile,strFile,basis_type,v1d,Hpot,njm,njmQuantumNumList,tol)

	Hrot=np.zeros((njm,njm),float)
	for jm in range(njm):
		for jmp in range(njm):
			if (jm==jmp):
				Hrot[jm,jm]=Bconst*njmQuantumNumList[jm,0]*(njmQuantumNumList[jm,0]+1.0)
    
	Htot = Hrot1 + Hpot
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
	exit()

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
