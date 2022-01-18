#********************************************************************************
#                                                                               | 
# Diagonalization code for a asymmetric top rotor system with real Wigner basis |
# set. See the Appendix of Ref: Rep. Prog. Phys. 77 (2014) 046601.              |
#                                                                               |
# It computes eigenvalues and eigenfunctions for a spinless system.             |
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
#       a) Distance between two COMs along z-axis = zcom                        |
#       b) Highest value of Angular quantum number = Jmax                       |
#       c) Specification of spin isomer = spin_isomer#                          |
#                                                                               |
#                              or                                               |
#                                                                               |
# Run "python diag_monomer_nonlinear_rot_real_basis.py --help" on the terminal  |
#                                                                               |
#-----------------------------------------------------------------------------  |
#                                                                               |
# Outputs: a) Eigenvalues and eigenfunctions                                    |
#          b) Expectation values of energies                                    |
#          c) Ground state angular (theta) distribution                         |
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

if __name__ == '__main__':    

	parser = argparse.ArgumentParser(prog="diag_monomer_nonlinear_rot_real_basis.py",description="Diagonalization code for a nonlinear rotor system  with real basis. See the Appendix of Ref: Rep. Prog. Phys. 77 (2014) 046601. IMPORTANT note: It is implemented only for spinless nuclear-spin isomer.",epilog="Enjoy the program! :)")
	parser.add_argument("zcom", help="Distance between two centre of masses along z-axix. It is a real number.")
	parser.add_argument("jmax", help="Truncated angular quantum number for this computation. It must be a non-negative integer number.")
	parser.add_argument("spin", help="It includes nuclear spin isomerism. It is a string.", choices=["para", "ortho", "spinless"])
	args = parser.parse_args()


	zCOM=float(args.zcom)             # Distance between two centre of masses along z-axis
	Jmax=int(args.jmax)               # Truncated angular quantum number for this computation
	spin_isomer=args.spin             # It includes nuclear spin isomerism

	assert ('spinless' in spin_isomer), "This code is implemented only for spinless system."

	size_theta = int(2*Jmax+3)
	size_phi = int(2*(2*Jmax+1))

	small = 10e-8                   # Tollerance to be considered a matrix hermitian of unitary

	#Printing conditions
	norm_check_real = True
	norm_check_complex = False
	io_write = True
	pot_write = False

	if (io_write == True):
		print("")
		print("")
		print(" Jmax = ", Jmax)
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

	strFile = "diag-2"+isomer+"H2O-one-rotor-fixed-cost-1-jmax"+str(Jmax)+"-Rpt"+str(zCOM)+"Angstrom-grids-"+str(size_theta)+"-"+str(size_phi)+"-real-wigner-basis.txt"
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
	njkmQuantumNumList_Real = bfunc.get_njkmQuantumNumList_RealBasis(Jmax,njkm)

	#Calling of real Wigner basis set <th, ph, ch | JKM> 
	# Its shape - (njkm,size_theta*size_phi*size_phi) 
	wigner_real = bfunc.get_NonLinear_RealBasis(Jmax,njkm,size_theta,size_phi,xGL,wGL,phixiGridPts,dphixi)
	if (norm_check_real == True):
		normMat_real = np.tensordot(wigner_real, wigner_real, axes=([1],[1]))
		bfunc.test_norm_NonLinear_RealBasis(prefile,strFile,basis_type,normMat_real,njkm,small)

	njkmQuantumNumList_Comp = bfunc.get_njkmQuantumNumList_NonLinear_ComplexBasis(njkm,Jmax,spin_isomer)
	dJKM, KJKM, MJKM = bfunc.get_wigner_ComplexBasis(njkm,size_theta,size_phi,njkmQuantumNumList_Comp,xGL,wGL,phixiGridPts,dphixi)

	#block for construction of |JKM> basis begins 
	wigner_complex1 = KJKM[:,np.newaxis,np.newaxis,:]*MJKM[:,np.newaxis,:,np.newaxis]*dJKM[:,:,np.newaxis,np.newaxis]
	wigner_complex = np.reshape(wigner_complex1,(njkm,size_theta*size_phi*size_phi),order='C')
	#block for construction of |JKM> basis ends

	if (norm_check_complex == True):
		normMat_complex = np.tensordot(np.conjugate(wigner_complex), wigner_complex, axes=([1],[1]))
		bfunc.test_norm_NonLinear_ComplexBasis(prefile,strFile,basis_type,normMat_complex,njkm,njkmQuantumNumList_Comp,small)


	# Construction of unitary matrix for the transformation between complex and real Wigner bases.
	#                                                                                           |
	#	<x_i|U|x_j> = <x_i|t_j> = int_Omega <x_i|Omega><Omega|t_j>                              |
	#																							|
	#********************************************************************************************
	umat = np.tensordot(np.conjugate(wigner_complex), wigner_real, axes=([1],[1]))

	print("")
	print("")
	print("|------------------------------------------------")
	print("")
	print("# Checking the matrix to be Unitary.")
	# Checknig if unitary condition is satisfied?
	uu = np.tensordot(np.conjugate(umat), umat, axes=([1],[1]))

	print("")
	print("# Real parts of diagonal elements of U^U matxix")
	print(uu.diagonal().real)
	print("")
	print("# Imaginary parts of diagonal elements of U^U matxix")
	print(uu.diagonal().imag)
	print("")
	print("")

	Hrot = bfunc.get_rotmat_NonLinear_ComplexBasis(njkm,njkmQuantumNumList_Comp,Ah2o,Bh2o,Ch2o)
	hu = np.tensordot(Hrot, umat, axes=([1],[0]))
	uhu = np.tensordot(np.conjugate(umat.T), hu,axes=([1],[0]))

	if (np.all(np.abs(uhu - uhu.T) < small) == False):
		print("|------------------------------------------------")
		print("Warning!!!!!!!!!!")
		print("")
		print("The Hamiltonian matrx uhu is not hermitian.")
		exit()

	if (np.any(np.abs(uhu.imag)>small)):
		print("")
		print("")
		print("|------------------------------------------------")
		print("Warning!!!!!!!!!!")
		print("")
		print("Non-real for rot matrix in real basis")
		exit()

	pot_func = bfunc.get_pot(size_theta,size_phi,zCOM,xGL,phixiGridPts)

	tempa = pot_func[np.newaxis,:]*wigner_real
	Hpot = np.tensordot(wigner_real, tempa, axes=([1],[1]))

	#if (pot_write == True):
	#	get_norm(prefile,strFile,basis_type,v1d,eEEebasisuse,Hpot,njkm,njkmQuantumNumList,small)

	Htot = uhu.real + Hpot

	#Estimation of eigenvalues and eigenvectors begins here
	eigVal, eigVec = LA.eigh(Htot)
	sortIndex_eigVal = eigVal.argsort()    
	eigVal_sort = eigVal[sortIndex_eigVal]       
	eigVec_sort = eigVec[:,sortIndex_eigVal]       
	#Estimation of eigenvalues and eigenvectors ends here

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
		avgHpot = np.dot(eigVec_sort[:,idx].T,avgHpotL)
		print("# {state:^10d} {eigval_re:^15.6f} {eigval_im:^15.6e}".format(state=idx,eigval_re=avgHpot.real,eigval_im=avgHpot.imag))
	# printing block is closed


	# printing block is opened
	idx=0
	avgHpotL = np.dot(Hpot,eigVec_sort[:,idx])
	avgHpot = np.dot(eigVec_sort[:,idx].T,avgHpotL)

	gs_eng_file = prefile+"ground-state-energy-"+strFile
	gs_eng_write = open(gs_eng_file,'w')
	gs_eng_write.write("# Printing of ground state energies in inverse Kelvin - "+"\n")
	gs_eng_write.write('{0:1} {1:^19} {2:^20}'.format("#","<T+V>", "<V>"))
	gs_eng_write.write("\n")
	gs_eng_write.write('{0:^20.8f} {1:^20.8f}'.format(eigVal_sort[0], avgHpot.real))
	gs_eng_write.write("\n")
	gs_eng_write.close()
	# printing block is closed

	# Computation of <JKM (complex basis) | JKM (Real basis)> for the estimation of reduced density matrix
	vec0_real = eigVec_sort[:,0]          # <JKM (Real basis) | 0>
	temp0     = vec0_real[np.newaxis,:]   # represent in the matrix form
	# <0|nc> = sum_nr <0|nr><nr|nc> ; nr ---> |JKM (Real basis)> & nc---> |JKM (Complex basis)>
	temp1     = np.tensordot(temp0, np.conjugate(umat), axes=([1],[1])) 
	vec0_comp = temp1[0,:]
	bfunc.get_1dtheta_distribution(Jmax,njkm,njkmQuantumNumList_Comp,vec0_comp,size_theta,xGL,wGL,dJKM,prefile,strFile)

	print("")
	print("")
	print("Successful execution!")
