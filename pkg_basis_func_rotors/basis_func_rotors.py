#************************************************************|
#															|
# This module is used for calling various basis functions	|
# for a linear and nonlinear top systems.					|
#															|
# It gives us complex spherical harmonics, real spherical	|
# harmonics and Wigner basis function.					   |
#															|
#************************************************************|

import sys
import math
import numpy as np
from scipy import special as sp
from scipy import linalg as LA
import pkg_potential.qpot as qpot
from scipy.sparse.linalg import eigs, eigsh
import cmath
import functools

def off_diag (j,k):						
	"""
	It computes off diagonal <JKM|H|J'K'M'> 
	"""
	f = np.sqrt((j*(j+1)) - (k*(k+1)))	 
	return f							   

def littleD(ldJ,ldmp,ldm,ldtheta):
	"""
	It Computes d(J, M', M, theta) ie. little d-rotation matrix 
	See the page no 86 of "Angular momentum - written by R N Zare"
	"""
	teza =(np.math.factorial(ldJ+ldm)*np.math.factorial(ldJ-ldm)*np.math.factorial(ldJ+ldmp)*np.math.factorial(ldJ-ldmp))*1.0
	dval = np.sqrt(teza) 
	tempD = 0.

	#determine max v that will begin to give negative factorial arguments
	if ((ldJ-ldmp) > (ldJ+ldm)):
		upper = ldJ-ldmp
	else:
		upper = ldJ+ldm

	#iterate over intergers that provide non-negative factorial arguments
	for v in range(upper+1):
		a = ldJ - ldmp - v
		b = ldJ + ldm - v
		c = v + ldmp - ldm
		if ((a>=0) and (b>=0) and (c>=0)):
			tempD=tempD+(((-1.0)**v)/(np.math.factorial(a)*np.math.factorial(b)*np.math.factorial(c)*np.math.factorial(v)))*((np.cos(ldtheta/2.))**(2.*ldJ+ldm-ldmp-2.*v))*((-np.sin(ldtheta/2.))**(ldmp-ldm+2.*v))
	return dval*tempD

def get_wigner_ComplexBasis(njkm,size_theta,size_phi,njkmQuantumNumList,xGL,wGL,phixiGridPts,dphixi):

	"""
	It construncts Wigner basis <theta, phi, chi | JKM>
	"""

	dJKM = np.zeros((njkm,size_theta),float)
	KJKM = np.zeros((njkm,size_phi),complex)
	MJKM = np.zeros((njkm,size_phi),complex)

	# Computation littleD(j,m,k,theta) and compare it with the date estimated by asymrho.f
	"""
	theta = 1.0 # in degree
	for s in range(njkm):
		print("j=",njkmQuantumNumList[s,0],"m=",njkmQuantumNumList[s,2],"k=",njkmQuantumNumList[s,1],littleD(njkmQuantumNumList[s,0],njkmQuantumNumList[s,2],njkmQuantumNumList[s,1],theta*np.pi/180.))
	"""

	Nk = 1.0
	for s in range(njkm):
		for th in range(size_theta):
			dJKM[s,th] = np.sqrt((2.*njkmQuantumNumList[s,0]+1)/(8.*np.pi**2))*littleD(njkmQuantumNumList[s,0],njkmQuantumNumList[s,2],njkmQuantumNumList[s,1],np.arccos(xGL[th]))*np.sqrt(wGL[th])

		for ph in range(size_phi):
			KJKM[s,ph] = np.exp(1j*phixiGridPts[ph]*njkmQuantumNumList[s,1])*np.sqrt(dphixi)*Nk
			MJKM[s,ph] = np.exp(1j*phixiGridPts[ph]*njkmQuantumNumList[s,2])*np.sqrt(dphixi)*Nk

	return dJKM, KJKM, MJKM

def get_njkmQuantumNumList_NonLinear_ComplexBasis(njkm,Jmax,spin_isomer):
	"""
	Lists of (J,K,M) quantum number indices computed for nuclear spin isomers

	Para isomer is obtained by summing over even K,
	Ortho isomer is obtained by summing over odd K,
	spinless is computed by summing over all K values.
	"""

	if (spin_isomer == "spinless"):
		JKM=njkm
		JKMQuantumNumList = np.zeros((JKM,3),int)
		#all J
		jtempcounter = 0
		for J in range(Jmax+1):
			for K in range(-J,J+1,1):
				for M in range(-J,J+1):
					JKMQuantumNumList[jtempcounter,0]=J
					JKMQuantumNumList[jtempcounter,1]=K
					JKMQuantumNumList[jtempcounter,2]=M
					jtempcounter+=1
		return JKMQuantumNumList

	if (spin_isomer == "para"):
		JKeM=njkm
		JKeMQuantumNumList = np.zeros((JKeM,3),int)
		#even
		jtempcounter = 0
		for J in range(Jmax+1):
			if ((J%2) == 0):
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
		return JKeMQuantumNumList

	if (spin_isomer == "ortho"):
		JKoM=njkm
		JKoMQuantumNumList = np.zeros((JKoM,3),int)
		#odd
		jtempcounter = 0
		for J in range(Jmax+1):
			if ((J%2) == 0):
				for K in range(-J+1,J,2):
					for M in range(-J,J+1):
						JKoMQuantumNumList[jtempcounter,0]=J
						JKoMQuantumNumList[jtempcounter,1]=K
						JKoMQuantumNumList[jtempcounter,2]=M
						jtempcounter+=1
			else:
				for K in range(-J,J+1,2):
					for M in range(-J,J+1):
						JKoMQuantumNumList[jtempcounter,0]=J
						JKoMQuantumNumList[jtempcounter,1]=K
						JKoMQuantumNumList[jtempcounter,2]=M
						jtempcounter+=1

		return JKoMQuantumNumList

def get_rotmat_NonLinear_ComplexBasis(njkm,njkmQuantumNumList,Ah2o,Bh2o,Ch2o):

	"""
	Construction of kinetic energy matrix
	The equations are taken from page 272 of the book ``Anular Momentum'' written by R. N. Zare

	The expression given in the book is for prolate top limit. Here we have used the 
	expressions for the intermediate situation (See the (Table 6.2 of Angular Momentum
	by zare)).
	"""

	Hrot = np.zeros((njkm,njkm),dtype=float)
	
	for jkm in range(njkm):
		for jkmp in range(njkm):
			if ((njkmQuantumNumList[jkm,0] == njkmQuantumNumList[jkmp,0]) and (njkmQuantumNumList[jkm,2] == njkmQuantumNumList[jkmp,2])):
				if (njkmQuantumNumList[jkm,1] == (njkmQuantumNumList[jkmp,1]-2)):
					Hrot[jkm,jkmp] += 0.25*(Ah2o-Ch2o)*off_diag(njkmQuantumNumList[jkm,0],njkmQuantumNumList[jkm,1])*off_diag(njkmQuantumNumList[jkm,0],njkmQuantumNumList[jkm,1]+1)
				elif (njkmQuantumNumList[jkm,1] == (njkmQuantumNumList[jkmp,1]+2)):
					Hrot[jkm,jkmp] += 0.25*(Ah2o-Ch2o)*off_diag(njkmQuantumNumList[jkm,0],njkmQuantumNumList[jkm,1]-1)*off_diag(njkmQuantumNumList[jkm,0],njkmQuantumNumList[jkm,1]-2)
				elif (njkmQuantumNumList[jkm,1] == (njkmQuantumNumList[jkmp,1])):
					Hrot[jkm,jkmp] += (0.5*(Ah2o + Ch2o)*(njkmQuantumNumList[jkm,0]*(njkmQuantumNumList[jkm,0]+1)) + (Bh2o - 0.5*(Ah2o+Ch2o)) * ((njkmQuantumNumList[jkm,1])**2))

	return Hrot

def get_pot(size_theta,size_phi,val,xGL,phixiGridPts):

	"""
	It computes potential matrix over the position basis.
	"""

	'''
	v1d = np.zeros(size_theta*size_phi*size_phi,float)
	ii = 0
	for th1 in range(size_theta):
		for ph1 in range(size_phi):
			for ch1 in range(size_phi):
				v1d[ii]=-zCOM*xGL[th1]
				v1d[ii]=0.0#-zCOM*xGL[th1]
				ii = ii + 1
	'''
	com1=[0.0,0.0,0.0]
	com2=[0.0,0.0,val]
	#Eulang2=[0.0,0.0,0.0] 
	Eulang2=[0.0, math.pi, 0.0]
	pot_func = np.zeros(size_theta*size_phi*size_phi,float)
	ii = 0
	for th1 in range(size_theta):
		for ph1 in range(size_phi):
			for ch1 in range(size_phi):
				ii = ch1+(ph1+th1*size_phi)*size_phi
				Eulang1=[phixiGridPts[ph1], math.acos(xGL[th1]), phixiGridPts[ch1]]
				pot_func[ii]=qpot.caleng(com1,com2,Eulang1,Eulang2)

	return pot_func

def get_wigner_RealBasis(njkm_J,njkm_K,njkm_M,theta,wt,phi,wp,chi,wc):
	"""
	See ``Appendix: Real basis of non-linear rotor'' in Rep. Prog. Phys. vol. 77 page- 046601 (2014).
	"""

	theta0 = math.sqrt((2.*njkm_J+1)/(8.*math.pi**2))*littleD(njkm_J,0,0,np.arccos(theta))*math.sqrt(wt)*math.sqrt(wp)*math.sqrt(wc)
	dd = math.sqrt((2.*njkm_J+1)/(4.*math.pi**2))*littleD(njkm_J,njkm_M,njkm_K,np.arccos(theta))*math.sqrt(wt)
	thetac = dd*math.cos(phi*njkm_M+chi*njkm_K)*math.sqrt(wp)*math.sqrt(wc)
	thetas = dd*math.sin(phi*njkm_M+chi*njkm_K)*math.sqrt(wp)*math.sqrt(wc)

	return theta0,thetac,thetas

def get_NonLinear_RealBasis(Jmax,njkm,size_theta,size_phi,xGL,wGL,phixiGridPts,dphixi):
	"""
	See ``Appendix: Real basis of non-linear rotor'' in Rep. Prog. Phys. vol. 77 page- 046601 (2014).
	"""

	basisf = np.zeros((njkm,size_theta*size_phi*size_phi),dtype=float)
			
	for th in range(size_theta):
		theta=xGL[th]
		wt=wGL[th]
		for ph in range(size_phi):
			phi=phixiGridPts[ph]
			wp=dphixi
			itp=ph+th*size_phi
			for ch in range(size_phi):
				chi=phixiGridPts[ch]
				wc=dphixi
				itpc=ch+itp*size_phi

				ib=0
				for J in range(Jmax+1):
					K=0

					M=0
					theta0,thetac,thetas = get_wigner_RealBasis(J,K,M,theta,wt,phi,wp,chi,wc)
					basisf[ib,itpc]=theta0
					ib=ib+1

					for M in range(1,J+1,1):
						theta0,thetac,thetas = get_wigner_RealBasis(J,K,M,theta,wt,phi,wp,chi,wc)
						basisf[ib,itpc]=thetac
						ib=ib+1
						basisf[ib,itpc]=thetas
						ib=ib+1
						
					for K in range(1,J+1):
						for M in range(-J,J+1,1):
							theta0,thetac,thetas = get_wigner_RealBasis(J,K,M,theta,wt,phi,wp,chi,wc)
							basisf[ib,itpc]=thetac
							ib=ib+1
							basisf[ib,itpc]=thetas
							ib=ib+1
					
	
	return basisf

def test_norm_NonLinear_RealBasis(prefile,strFile,basis_type,normMat,njkm,small):
	"""
	It checks if the real wigner basis functions are normalized?
	"""
	fname = prefile+"norm-check-"+strFile
	fwrite = open(fname,'w')
	fwrite.write("#*******************************************************\n")
	fwrite.write("\n")
	fwrite.write("# Normalization conditions for Real Wigner basis set.\n")
	fwrite.write("\n")

	fwrite.write("# normMat.shape: shape of the "+basis_type+" <JKM|JKM> basis: " + str(normMat.shape)+" \n")
	fwrite.write("\n")
	fwrite.write("\n")

	for s1 in range(njkm):
		for s2 in range(njkm):
			if (np.abs(normMat[s1,s2]) > small):
				fwrite.write("L vec Rotor1: "+str(s1)+"\n")
				fwrite.write("R vec Rotor1: "+str(s2)+"\n")
				fwrite.write("Norm: "+str(normMat[s1,s2])+"\n")
				fwrite.write("\n")
	fwrite.close()

def get_njkmQuantumNumList_RealBasis(Jmax,njkm):
	"""
	See ``Appendix: Real basis of non-linear rotor'' in Rep. Prog. Phys. vol. 77 page- 046601 (2014).
	"""
	JKMQuantumNumList = np.zeros((njkm,3),int)

	jtempcounter = 0
	for J in range(Jmax+1):
		K=0
		M=0

		JKMQuantumNumList[jtempcounter,0]=J
		JKMQuantumNumList[jtempcounter,1]=K
		JKMQuantumNumList[jtempcounter,2]=M
		jtempcounter=jtempcounter+1

		for M in range(1,J+1,1):

			JKMQuantumNumList[jtempcounter,0]=J
			JKMQuantumNumList[jtempcounter,1]=K
			JKMQuantumNumList[jtempcounter,2]=M
			jtempcounter=jtempcounter+1
			JKMQuantumNumList[jtempcounter,0]=J
			JKMQuantumNumList[jtempcounter,1]=K
			JKMQuantumNumList[jtempcounter,2]=M
			jtempcounter=jtempcounter+1

		for K in range(1,J+1):
			for M in range(-J,J+1,1):
				JKMQuantumNumList[jtempcounter,0]=J
				JKMQuantumNumList[jtempcounter,1]=K
				JKMQuantumNumList[jtempcounter,2]=M
				jtempcounter=jtempcounter+1
				JKMQuantumNumList[jtempcounter,0]=J
				JKMQuantumNumList[jtempcounter,1]=K
				JKMQuantumNumList[jtempcounter,2]=M
				jtempcounter=jtempcounter+1

	return JKMQuantumNumList

def test_norm_NonLinear_ComplexBasis(prefile,strFile,basis_type,normMat,njkm,njkmQuantumNumList,small):
	"""
	It is used to check if normalization condition is satisfied? 

	Here Dr. Sahoo use metaprogramming to modify the function output.
	"""
	fname = prefile+"norm-check-"+strFile
	fwrite = open(fname,'w')
	fwrite.write("#*******************************************************"+" \n")
	fwrite.write("\n")
	fwrite.write("# Normalization conditions for Complex Wigner basis set.\n")
	fwrite.write("\n")
	fwrite.write("# normMat.shape: shape of the "+basis_type+" <JKM|JKM> basis: " + str(normMat.shape)+" \n")
	fwrite.write("\n")
	fwrite.write("\n")

	for s1 in range(njkm):
		for s2 in range(njkm):
			if (np.abs(normMat[s1,s2]) > small):
				fwrite.write("L vec Rotor1: "+str(njkmQuantumNumList[s1,0])+" "+str(njkmQuantumNumList[s1,1])+" "+str(njkmQuantumNumList[s1,2])+"\n")
				fwrite.write("R vec Rotor1: "+str(njkmQuantumNumList[s2,0])+" "+str(njkmQuantumNumList[s2,1])+" "+str(njkmQuantumNumList[s2,2])+"\n")
				fwrite.write("Norm: "+str(normMat[s1,s2])+"\n")
				fwrite.write("\n")
	fwrite.close()

def generate_linear_rotor_quantum_numbers(num_basis_functions, max_angular_quantum_number, spin_isomer_type):
	"""
	Generates quantum numbers for a linear rotor system based on the spin isomer type:
	'spinless', 'para', or 'ortho'.
	
	Parameters:
	- num_basis_functions (int): Total number of basis functions.
	- max_angular_quantum_number (int): Maximum angular quantum number, Jmax.
	- spin_isomer_type (str): Type of spin isomer ('spinless' for all J values, 
							   'para' for even J values, 'ortho' for odd J values).
	
	Returns:
	- np.ndarray: A 2D array of quantum numbers with columns [J, M], 
				  limited by num_basis_functions if necessary.
	"""
	
	quantum_numbers = []
	counter = 0

	if spin_isomer_type == "spinless":
		for J in range(max_angular_quantum_number + 1):
			for M in range(-J, J + 1):
				quantum_numbers.append([J, M])
				counter += 1
				if counter >= num_basis_functions:
					return np.array(quantum_numbers)

	elif spin_isomer_type == "para":
		for J in range(0, max_angular_quantum_number + 1, 2):
			for M in range(-J, J + 1):
				quantum_numbers.append([J, M])
				counter += 1
				if counter >= num_basis_functions:
					return np.array(quantum_numbers)

	elif spin_isomer_type == "ortho":
		for J in range(1, max_angular_quantum_number + 1, 2):
			for M in range(-J, J + 1):
				quantum_numbers.append([J, M])
				counter += 1
				if counter >= num_basis_functions:
					return np.array(quantum_numbers)

	else:
		raise ValueError("Invalid spin isomer type. Choose 'spinless', 'para', or 'ortho'.")

	return np.array(quantum_numbers)


def check_basis_normalization(output_filename, basis_label, basis_matrix, normalization_matrix, num_quantum_states, quantum_numbers, tolerance):
	"""
	Validates the orthonormality condition for basis functions by checking if <JM|J'M'> = δ_JJ' δ_MM'.
	
	Parameters:
	- output_filename (str): Base filename for saving the normalization check results.
	- basis_label (str): Descriptive label for the basis type (e.g., "rotor" or "spherical harmonics").
	- basis_matrix (np.ndarray): Matrix of basis functions with shape (num_grid_points, num_basis_functions).
	- normalization_matrix (np.ndarray): Matrix (num_basis_functions, num_basis_functions) expected to be close to identity.
	- num_quantum_states (int): Number of unique (J, M) quantum states (basis functions).
	- quantum_numbers (np.ndarray): Array of quantum numbers with columns [J, M] corresponding to each basis function.
	- tolerance (float): Threshold for numerical error; values below this are considered zero.
	"""
	
	# Construct full output file path for normalization check results
	normalization_output_file = output_filename + f"_normalization_check_{basis_label}.txt"
	
	# Open file for writing results
	with open(normalization_output_file, 'w') as file:
		# Record the shapes of basis and normalization matrices
		file.write(f"Basis matrix shape for {basis_label} |JM>: {basis_matrix.shape}\n")
		file.write(f"Normalization matrix shape for {basis_label} <JM|JM>: {normalization_matrix.shape}\n\n")
		
		# Verify orthonormality condition for each pair of basis states
		for row_idx in range(num_quantum_states):
			for col_idx in range(num_quantum_states):
				# Check if off-diagonal terms are within the acceptable tolerance
				if np.abs(normalization_matrix[row_idx, col_idx]) > tolerance:
					# Retrieve quantum numbers for each state pair
					J_row, M_row = quantum_numbers[row_idx]
					J_col, M_col = quantum_numbers[col_idx]
					
					# Extract real and imaginary parts for easier reading
					real_part = np.real(normalization_matrix[row_idx, col_idx])
					imag_part = np.imag(normalization_matrix[row_idx, col_idx])
					
					# Write any deviations from orthonormality to the file
					file.write(f"Left State (J, M): ({J_row}, {M_row})\n")
					file.write(f"Right State (J, M): ({J_col}, {M_col})\n")
					file.write(f"Deviation in normalization - Real part: {real_part:.5f}, Imaginary part: {imag_part:.5f}\n\n")


def normalization_checkLinear(file_name_normalization,basis_type,eEEbasisuse,normMat,njm,njmQuantumNumList,small):
	"""
	Check normalization condition: <JM|J'M'>=delta_JJ'MM'
	"""
	norm_check_file = prefile+"norm-check-"+strFile
	norm_check_write = open(norm_check_file,'w')
	norm_check_write.write("eEEbasisuse.shape: shape of the "+basis_type+" |JM> basis: " + str(eEEbasisuse.shape)+" \n")
	norm_check_write.write("normMat.shape: shape of the "+basis_type+" <JM|JM> basis: " + str(normMat.shape)+" \n")
	norm_check_write.write("\n")
	norm_check_write.write("\n")

	for s1 in range(njm):
		for s2 in range(njm):
			if (np.abs(normMat[s1,s2]) > small):
				norm_check_write.write("L vec Rotor1: "+str(njmQuantumNumList[s1,0])+" "+str(njmQuantumNumList[s1,1])+"\n")
				norm_check_write.write("R vec Rotor1: "+str(njmQuantumNumList[s2,0])+" "+str(njmQuantumNumList[s2,1])+"\n")
				norm_check_write.write("Constant potential field - Re: "+str(np.real(normMat[s1,s2]))+"   Im: "+str(np.imag(normMat[s1,s2]))+"\n")
				norm_check_write.write("\n")
	norm_check_write.close()

def spherical_harmonicsReal(njm,size_theta,size_phi,njmQuantumNumList,xGL,wGL,phixiGridPts,dphixi):

	"""
	It constructs real spherical harmonics in terms of complex spherical harmonics.

	For the reference see spherical harmonics wikipedia page
	"""
	basisfun = np.zeros((size_theta*size_phi,njm),float)
	#basisfun = np.zeros((size_theta*size_phi,njm),complex)

	'''
	for s in range(njm):
		Nk1=np.sqrt(2.0)*(-1.0)**njmQuantumNumList[s,1]
		Nk2=(2.*njmQuantumNumList[s,0]+1.0)/(4.0*np.pi)
		Nk3=math.factorial(njmQuantumNumList[s,0]-abs(njmQuantumNumList[s,1]))/math.factorial(njmQuantumNumList[s,0]+abs(njmQuantumNumList[s,1]))
		for th in range(size_theta):
			for ph in range(size_phi):
				ii = ph+th*size_phi
			
				if (njmQuantumNumList[s,1] == 0):
					basisfun[ii,s] = np.sqrt(Nk2)*sp.lpmv(0,float(njmQuantumNumList[s,0]),xGL[th])*np.sqrt(wGL[th])*np.sqrt(dphixi)
				if (njmQuantumNumList[s,1] < 0):
					Nk=Nk1*np.sqrt(Nk2*Nk3)
					basisfun[ii,s] = Nk*sp.lpmv(abs(njmQuantumNumList[s,1]),float(njmQuantumNumList[s,0]),xGL[th])*np.sin(abs(njmQuantumNumList[s,0])*phixiGridPts[ph])*np.sqrt(wGL[th])*np.sqrt(dphixi)
				if (njmQuantumNumList[s,1] > 0):
					Nk=Nk1*np.sqrt(Nk2*Nk3)
					basisfun[ii,s] = Nk*sp.lpmv(njmQuantumNumList[s,1],float(njmQuantumNumList[s,0]),xGL[th])*np.cos(njmQuantumNumList[s,0]*phixiGridPts[ph])*np.sqrt(wGL[th])*np.sqrt(dphixi)
	'''

	'''
	for jm in range(njm):
		for th in range(size_theta):
			for ph in range(size_phi):
				ii = ph+th*size_phi
			
				if (njmQuantumNumList[jm,1] == 0):
					basisfun[ii,jm]=sp.sph_harm(njmQuantumNumList[jm,1],njmQuantumNumList[jm,0],phixiGridPts[ph],np.arccos(xGL[th]))*np.sqrt(wGL[th])*np.sqrt(dphixi)
				if (njmQuantumNumList[jm,1] < 0):
					Nk=1./np.sqrt(2.0)
					basisfun[ii,jm]=Nk*(sp.sph_harm(abs(njmQuantumNumList[jm,1]),njmQuantumNumList[jm,0],phixiGridPts[ph],np.arccos(xGL[th]))-1j*sp.sph_harm(-abs(njmQuantumNumList[jm,1]),njmQuantumNumList[jm,0],phixiGridPts[ph],np.arccos(xGL[th])))*np.sqrt(wGL[th])*np.sqrt(dphixi)
				if (njmQuantumNumList[jm,1] > 0):
					Nk=((-1.0)**njmQuantumNumList[jm,1])/np.sqrt(2.0)
					basisfun[ii,jm]=Nk*(sp.sph_harm(abs(njmQuantumNumList[jm,1]),njmQuantumNumList[jm,0],phixiGridPts[ph],np.arccos(xGL[th]))+1j*sp.sph_harm(-abs(njmQuantumNumList[jm,1]),njmQuantumNumList[jm,0],phixiGridPts[ph],np.arccos(xGL[th])))*np.sqrt(wGL[th])*np.sqrt(dphixi)
	'''

	for jm in range(njm):
		for th in range(size_theta):
			for ph in range(size_phi):
				ii = ph+th*size_phi
			
				if (njmQuantumNumList[jm,1] == 0):
					basisfun[ii,jm]=sp.sph_harm(njmQuantumNumList[jm,1],njmQuantumNumList[jm,0],phixiGridPts[ph],np.arccos(xGL[th])).real*np.sqrt(wGL[th])*np.sqrt(dphixi)
				if (njmQuantumNumList[jm,1] < 0):
					Nk=((-1.0)**njmQuantumNumList[jm,1])*np.sqrt(2.0)
					basisfun[ii,jm]=Nk*sp.sph_harm(abs(njmQuantumNumList[jm,1]),njmQuantumNumList[jm,0],phixiGridPts[ph],np.arccos(xGL[th])).imag*np.sqrt(wGL[th])*np.sqrt(dphixi)
				if (njmQuantumNumList[jm,1] > 0):
					Nk=((-1.0)**njmQuantumNumList[jm,1])*np.sqrt(2.0)
					#basisfun[ii,jm]=Nk*np.real(sp.sph_harm(njmQuantumNumList[jm,1],njmQuantumNumList[jm,0],phixiGridPts[ph],np.arccos(xGL[th])))*np.sqrt(wGL[th])*np.sqrt(dphixi)
					basisfun[ii,jm]=Nk*sp.sph_harm(njmQuantumNumList[jm,1],njmQuantumNumList[jm,0],phixiGridPts[ph],np.arccos(xGL[th])).real*np.sqrt(wGL[th])*np.sqrt(dphixi)
	

	return basisfun


def spherical_harmonicsComp(njm,size_theta,size_phi,njmQuantumNumList,xGL,wGL,phixiGridPts,dphixi):
	"""
	This function constructs commplex spherical harmonics for linear rotors, <theta, phi|lm>. 
	"""
	basisfun=np.zeros((size_theta*size_phi,njm),complex)
	for jm in range(njm):
		for th in range(size_theta):
			for ph in range(size_phi):
				ii = ph+th*size_phi
				basisfun[ii,jm]=sp.sph_harm(njmQuantumNumList[jm,1],njmQuantumNumList[jm,0],phixiGridPts[ph],np.arccos(xGL[th]))*np.sqrt(wGL[th])*np.sqrt(dphixi)
	return basisfun


def get_1dtheta_distribution(Jmax,njkm,njkmQuantumNumList,vec0,size_theta,xGL,wGL,dJKM,prefile,strFile):
	"""
	See the APPENDIX: DERIVATION OF THE ANGULAR DISTRIBUTION FUNCTION of J. Chem. Phys. 154, 244305 (2021).
	"""
	reduced_density=np.zeros((njkm,Jmax+1),dtype=complex)
	for i in range(njkm):
		for ip in range(njkm):
			if ((njkmQuantumNumList[i,1]==njkmQuantumNumList[ip,1]) and (njkmQuantumNumList[i,2]==njkmQuantumNumList[ip,2])):
				reduced_density[i,njkmQuantumNumList[ip,0]]=np.conjugate(vec0)[i]*vec0[ip]

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
