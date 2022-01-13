#********************************************************************************
#                                                                               | 
# Diagonalization code for a asymmetric top rotor system with Wigner basis set. |
#                                                                               |
# Developed by Dr. Tapas Sahoo                                                  |
#                                                                               |
#-------------------------------------------------------------------------------|
#                                                                               |
# Command for running the code:                                                 |
#                                                                               |
#       Example:                                                                |
#       python diag_monomer_nonlinear_rot_real_basis.py 10.0 2 0                |
#                                                                               |
#----------------------------------------------------------------------------   |
#                                                                               |
# Inputs: See first few lines of the code                                       |
#                                                                               |
#         a) Distance between two COMs along z-axis = zcom                      |
#         b) Highest value of Angular quantum number = Jmax                     |
#         c) Specification of spin isomer = spin_isomer#                        |
#                                                                               |
#                              or                                               |
#                                                                               |
# Run "python diag_monomer_nonlinear_rot_real_basis.py --help" on the terminal  |
#                                                                               |
#-----------------------------------------------------------------------------  |
#                                                                               |
# Outputs: a) Eigenvalues and eigenfunctions                                    |
#          b) Expectation values of energies                                    |
#          c) Ground state angular (theta) distribution                         !
#                                                                               |
#********************************************************************************

import argparse
import sys
import math
import numpy as np
from scipy import linalg as LA
from scipy.sparse.linalg import eigs, eigsh
import cmath

# Imports basis functions of rotors (linear and nonlinear rotors)
import pkg_basis_func_rotors.basis_func_rotors as bfunc 

# 'qpot' imports qTIP4P/Fw water model potential function
#import pkg_potential as qpot 

if __name__ == '__main__':    

	parser = argparse.ArgumentParser(prog="diag_monomer_nonlinear_rot_complex_basis.py",description="Diagonalization code for a nonlinear rotor system  with Wigner basis. See the book ``Angular Momentum'' written by R. N. Zare.",epilog="Enjoy the program! :)")
	parser.add_argument("zcom", help="Distance between two centre of masses along z-axix. It is a real number.")
	parser.add_argument("jmax", help="Truncated angular quantum number for this computation. It must be a non-negative integer number.")
	parser.add_argument("spin", help="It includes nuclear spin isomerism. It is a string.", choices=["para", "ortho", "spinless"])
	args = parser.parse_args()


	zCOM=float(args.zcom)             # Distance between two centre of masses along z-axis
	Jmax=int(args.jmax)               # Truncated angular quantum number for this computation
	spin_isomer=args.spin             # It includes nuclear spin isomerism

	size_theta = int(2*Jmax+3)
	size_phi = int(2*(2*Jmax+1))

	tol = 10e-8                       # Tollerance for checking if the matrix is hermitian?
	#Printing conditions
	norm_check = True
	io_write = False
	pot_write = False
	if (io_write == True):
		print("Jmax = ", Jmax)
		print(" Number of theta grids = ", size_theta)
		print(" Number of phi and chi grids = ", size_phi)
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

	strFile = "of-2"+isomer+"H2O-one-rotor-fixed-cost-1-jmax"+str(Jmax)+"-Rpt"+str(zCOM)+"Angstrom-grids-"+str(size_theta)+"-"+str(size_phi)+"-complex-wigner-basis.txt"
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
		print("")
		print("")
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
		print("")
		print("")
		print("|------------------------------------------------")
		print("| Number of basis functions calculations ....")
		print("| ")
		print("| Number of |JKM> basis = "+str(JKM))
		print("| ")
		print("| Number of even K in |JKM> = "+str(JKeM))
		print("| Number of odd  K in |JKM> = "+str(JKoM))
		print("| ")
		print("|------------------------------------------------")
		
	# Total number of |JKM> basis are determined as following -
	# for para, only even K values are considered,
	# for ortho, only odd K are included
	# and for spinless, all K values are added.
	if (spin_isomer == "spinless"):
		njkm = JKM	
	if (spin_isomer == "para"):
		njkm = JKeM	
	if (spin_isomer == "ortho"):
		njkm = JKoM	

	# List of (J,K,M) indices computed for various nuclear spin isomers
    # Its a 2-dim matrix
	njkmQuantumNumList = bfunc.get_njkmQuantumNumList_NonLinear_ComplexBasis(njkm,Jmax,spin_isomer)

	dJKM, KJKM, MJKM = bfunc.get_wigner_ComplexBasis(njkm,size_theta,size_phi,njkmQuantumNumList,xGL,wGL,phixiGridPts,dphixi)

	#Construction of complex Wigner basis set <th, ph, ch | JKM> 
	# Its shape - (njkm,size_theta*size_phi*size_phi) 
	basis_func = KJKM[:,np.newaxis,np.newaxis,:]*MJKM[:,np.newaxis,:,np.newaxis]*dJKM[:,:,np.newaxis,np.newaxis]
	basis_func = np.reshape(basis_func,(njkm,size_theta*size_phi*size_phi),order='C')

	if (norm_check == True):
		normMat = np.tensordot(np.conjugate(basis_func), basis_func, axes=([1],[1]))
		bfunc.test_norm_NonLinear_ComplexBasis(prefile,strFile,basis_type,normMat,njkm,njkmQuantumNumList,tol)

	pot_func = bfunc.get_pot(size_theta,size_phi,zCOM,xGL,phixiGridPts)

	tempa = pot_func[np.newaxis,:]*basis_func
	Hpot = np.tensordot(np.conjugate(basis_func), tempa, axes=([1],[1]))

	if (pot_write == True):
		bfunc.test_norm_NonLinear_ComplexBasis(prefile,strFile,basis_type,v1d,njkm,njkmQuantumNumList,tol)

	Hrot = bfunc.get_rotmat_NonLinear_ComplexBasis(njkm,njkmQuantumNumList,Ah2o,Bh2o,Ch2o)
    
	Htot = Hrot# + Hpot
	if (np.all(np.abs(Htot-Htot.T) < tol) == False):
		print("The Hamiltonian matrx Htot is not hermitian.")
		exit()

	# Estimation of eigenvalues and eigenvectors begins here
	eigVal, eigVec = LA.eigh(Htot)

	# Sorting the eigenvalues and placement of  the corresponding eigenvectors accordingly
	sortIndex_eigVal = eigVal.argsort()     
	eigVal_sort = eigVal[sortIndex_eigVal]       
	eigVec_sort = eigVec[:,sortIndex_eigVal]       
	# Estimation of eigenvalues and eigenvectors ends here

	#printing block is opened
	eigVal_comb = np.array([eigVal_sort, eigVal_sort/CMRECIP2KL])

	eigVal_file = prefile+"eigen-values-"+strFile
	np.savetxt(eigVal_file, eigVal_comb.T, fmt='%20.8f', delimiter=' ', header='Energy levels of a aymmetric top - Units associated with the first and second columns are Kelvin and wavenumber, respectively. ')

	# Printing for the checking of normalization conditions and computations of <V> for first FOUR eigenvectors
	print("")
	print("")
	print("#***************************************************************************************")
	print("")
	print("# Checking normalizations of first four eigenfunctions - ")
	print("")
	print("# {state:^10s} {eigval_re:^15s} {eigval_im:^15s}".format(state="states",eigval_re="Norm.Real", eigval_im="Norm.Imag"))
	for idx in range(4):
		eigVecRe = np.real(np.dot(np.conjugate(eigVec_sort[:,idx].T),eigVec_sort[:,idx])) # Real part of the Norm
		eigVecIm = np.imag(np.dot(np.conjugate(eigVec_sort[:,idx].T),eigVec_sort[:,idx])) # Imaginary part of the Norm
		print("# {state:^10d} {eigval_re:^15.6e} {eigval_im:^15.6e}".format(state=idx,eigval_re=eigVecRe,eigval_im=eigVecIm))

	print("")
	print("")
	print("#***************************************************************************************")
	print("")
	print("# Expectation values of potential energy for first four states - ")
	print("")
	print("# {state:^10s} {eigval_re:^15s} {eigval_im:^15s}".format(state="states",eigval_re="<V>.Real", eigval_im="<V>.Imag"))
	for idx in range(4):
		avgHpotL = np.dot(Hpot,eigVec_sort[:,idx])
		avgHpot = np.dot(np.conjugate(eigVec_sort[:,idx].T),avgHpotL)
		print("# {state:^10d} {eigval_re:^15.6f} {eigval_im:^15.6e}".format(state=idx,eigval_re=avgHpot.real,eigval_im=avgHpot.imag))
	# printing block is closed

	# printing block is opened
	idx=0
	avgHpotL = np.dot(Hpot,eigVec_sort[:,idx])
	avgHpot = np.dot(np.conjugate(eigVec_sort[:,idx].T),avgHpotL)

	gs_eng_file = prefile+"ground-state-energy-"+strFile
	gs_eng_write = open(gs_eng_file,'w')
	gs_eng_write.write("# Printing of ground state energies in inverse Kelvin - "+"\n")
	gs_eng_write.write('{0:1} {1:^19} {2:^20}'.format("#","<T+V>", "<V>"))
	gs_eng_write.write("\n")
	gs_eng_write.write('{0:^20.8f} {1:^20.8f}'.format(eigVal_sort[0], avgHpot.real))
	gs_eng_write.write("\n")
	gs_eng_write.close()
	# printing block is closed

	# computation of reduced density matrix
	#
	# See the APPENDIX: DERIVATION OF THE ANGULAR DISTRIBUTION FUNCTION of J. Chem. Phys. 154, 244305 (2021).
	#
	reduced_density=np.zeros((njkm,Jmax+1),dtype=complex)
	for i in range(njkm):
		for ip in range(njkm):
			if ((njkmQuantumNumList[i,1]==njkmQuantumNumList[ip,1]) and (njkmQuantumNumList[i,2]==njkmQuantumNumList[ip,2])):
				reduced_density[i,njkmQuantumNumList[ip,0]]=np.conjugate(eigVec_sort[i,0])*eigVec_sort[ip,0]

	gs_ang_file = prefile+"ground-state-theta-distribution-"+strFile
	gs_ang_write = open(gs_ang_file,'w')
	gs_ang_write.write("#Printing of ground state theta distribution - "+"\n")
	gs_ang_write.write('{0:1} {1:^19} {2:^20}'.format("#","cos(theta)", "Reduced density"))
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
 	
	print("")
	print("")
	print("#***************************************************************************************")
	print("")
	print("Normalization of the wavefunction that is used to compute reduced density matrix = ",sum3)
