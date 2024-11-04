#******************************************************************************
#                                                                             | 
# Diagonalization code for linear rotor system  with real spherical harmonics |
#                                                                             |
# Developed by Dr. Tapas Sahoo                                                |
#                                                                             |
#---------------------------------------------------------------------------- |
#                                                                             |
# Command for running the code:                                               |
#                                                                             |
#                                                                             |
#       Example:                                                              |
#       python diag_monomer_linear_rot_real_basis.py 10.0 2 0                 |
#                                                                             |
#---------------------------------------------------------------------------- |
#                                                                             |
# Inputs: See first few lines of the code                                     |
#                                                                             |
#       a) Potential strength = strength                                      |
#       b) Highest value of Angular quantum number = Jmax                     |
#       c) Specification of spin isomer = spin_isomer#                        |
#                                                                             |
#-----------------------------------------------------------------------------|
#                                                                             |
# Outputs: Eigenvalues and eigenfunctions                                     |
#                                                                             |
#*****************************************************************************

import argparse, sys, math, termcolor 
import numpy as np
import scipy
from scipy import linalg as LA
from scipy.sparse.linalg import eigs, eigsh
import cmath
from datetime import datetime
from termcolor import colored

# Imports basis functions of rotors (linear and nonlinear rotors)
import pkg_basis_func_rotors.basis_func_rotors as bfunc 

# 'qpot' imports qTIP4P/Fw water model potential function                                 
import pkg_potential as qpot 


if __name__ == '__main__':    

	parser = argparse.ArgumentParser(prog="diag_monomer_linear_rot_real_basis.py",description="Diagonalization code for linear rotor system  with real spherical harmonics.",epilog="Enjoy the program! :)")
	parser.add_argument("strength", help="It determines interaction strength of the potential form A*cos(theta). It is a real number.")
	parser.add_argument("jmax", help="Truncated angular quantum number for this computation. It must be a non-negative integer number.")
	parser.add_argument("spin", help="It includes nuclear spin isomerism. It is a string.", choices=["para", "ortho", "spinless"])
	args = parser.parse_args()

    # Potentail strength A: A*cos(theta)
	strength=args.strength 
	Jmax=int(args.jmax)

    # spin isomers: 
    # For para, ortho and spinless systems set it 0, 1, -1, separately.
	spin_isomer = args.spin
    # No. of grid points along theta and phi
	size_theta = int(2*Jmax+5)
	size_phi = int(2*(2*Jmax+5))

    # Tolerance limit for a harmitian matrix
	tol = 10e-8

	print("*"*80 + "\n")
	print(colored("Developer:".ljust(30),"blue") + colored("Dr. Tapas Sahoo", "yellow") + "\n")
	now = datetime.now() # current date and time
	date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
	print("date and time:".capitalize().ljust(29), date_time, "\n")
	print("*"*80 + "\n")
	exit()

	debugging=False
	if debugging:
		print(colored("File systems are given below:", "blue") + "\n")
		print("user_name: ".ljust(30) + user_name)
		print("home: ".ljust(30) + home)
		print("input_dir_path: ".ljust(30) + input_dir_path)

	#print the normalization 
	io_write = False
	norm_check = False
	pot_write = False

	if (io_write == True):
		print("Jmax = ", Jmax, flush=True)
		print(" Number of theta grids = ", size_theta, flush=True)
		print(" Number of phi and chi grids = ", size_phi, flush=True)
		sys.stdout.flush()

    # Below are specified for naming the output files
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

	Bconst = 60.853                 #cm-1 Taken from NIST data https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=1000
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
		
	# Total number of |JM> basis are -
	# for para, only even J values are considered,
	# for ortho, only odd J are included
	# and for spinless, all J values are added.
	if (spin_isomer == "spinless"):
		njm = JM	
	if (spin_isomer == "para"):
		njm = JeM	
	if (spin_isomer == "ortho"):
		njm = JoM	

	# List of (J,M) indices computed for various nuclear spin isomers
    # Its a 2-dim matrix
	njmQuantumNumList = bfunc.get_numbbasisLinear(njm,Jmax,spin_isomer)

    # Real spherical harmonics < cos(theta), phi | JM>
    # basisfun is a 2-dim matrix (size_theta*size_phi, njm)
	basisfun=bfunc.spherical_harmonicsReal(njm,size_theta,size_phi,njmQuantumNumList,xGL,wGL,phixiGridPts,dphixi)

	if (norm_check == True):
        # Dimension of normMat is (njm, njm)
		normMat = np.tensordot(basisfun, np.conjugate(basisfun), axes=([0],[0]))
        # Below the function checks normalization condition <lm|l'm'>=delta_ll'mm'
		bfunc.normalization_checkLinear(prefile,strFile,basis_type,basisfun,normMat,njm,njmQuantumNumList,tol)

	basisfun1=bfunc.spherical_harmonicsComp(njm,size_theta,size_phi,njmQuantumNumList,xGL,wGL,phixiGridPts,dphixi)
	normMat1 = np.tensordot(basisfun, basisfun1, axes=([0],[0]))

    # Computation of rotational kinetic energy operator in (lm) basis: H(lm,lm)
	Hrot1=np.zeros((njm,njm),float)
	for jm in range(njm):
		for jmp in range(njm):
			sum=0.0
			for s in range(njm):
				sum+=np.real(normMat1[jm,s]*np.conjugate(normMat1[jmp,s]))*Bconst*njmQuantumNumList[s,0]*(njmQuantumNumList[s,0]+1.0)
			Hrot1[jm,jmp]=sum


    # Computation of potential energy operator in (lm) basis: H(lm,lm)
	v1d=np.zeros((size_theta*size_phi),float)
	for th in range(size_theta):
		for ph in range(size_phi):
			v1d[ph+th*size_phi] = -strength*xGL[th] #A*cos(theta)

	tempa = v1d[:,np.newaxis]*basisfun
	Hpot = np.tensordot(np.conjugate(basisfun), tempa, axes=([0],[0]))

	if (pot_write == True):
		bfunc.normalization_checkLinear(prefile,strFile,basis_type,v1d,Hpot,njm,njmQuantumNumList,tol)

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
