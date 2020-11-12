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

def wigner_basis(njkm,size_theta,size_phi,njkmQuantumNumList,littleD,xGL,wGL,phixiGridPts,dphixi):

	'''
	construnction of wigner basis
	'''

	dJKM = np.zeros((njkm,size_theta),float)
	KJKM = np.zeros((njkm,size_phi),complex)
	MJKM = np.zeros((njkm,size_phi),complex)

	# Compute littleD(j,m,k,theta) and compare it with the date estimated by asymrho.f
	'''
	theta = 1.0 # in degree
	for s in range(njkm):
		print("j=",njkmQuantumNumList[s,0],"m=",njkmQuantumNumList[s,2],"k=",njkmQuantumNumList[s,1],littleD(njkmQuantumNumList[s,0],njkmQuantumNumList[s,2],njkmQuantumNumList[s,1],theta*np.pi/180.))
	'''

	Nk = 1.0
	for s in range(njkm):
		for th in range(size_theta):
			dJKM[s,th] = np.sqrt((2.*njkmQuantumNumList[s,0]+1)/(8.*np.pi**2))*littleD(njkmQuantumNumList[s,0],njkmQuantumNumList[s,2],njkmQuantumNumList[s,1],np.arccos(xGL[th]))*np.sqrt(wGL[th])

		for ph in range(size_phi):
			KJKM[s,ph] = np.exp(1j*phixiGridPts[ph]*njkmQuantumNumList[s,1])*np.sqrt(dphixi)*Nk
			MJKM[s,ph] = np.exp(1j*phixiGridPts[ph]*njkmQuantumNumList[s,2])*np.sqrt(dphixi)*Nk

	return dJKM, KJKM, MJKM

def norm_wigner(prefile,strFile,basis_type,eEEbasisuse,eEEebasisuse,normMat,njkm,njkmQuantumNumList,tol):
	norm_check_file = prefile+"norm-check-"+strFile
	norm_check_write = open(norm_check_file,'w')
	norm_check_write.write("eEEbasisuse.shape: shape of the "+basis_type+" |JKM> basis: " + str(eEEbasisuse.shape)+" \n")
	norm_check_write.write("eEEebasisuse.shape: reduced shape of the "+basis_type+" |JKM> basis: " + str(eEEebasisuse.shape)+" \n")
	norm_check_write.write("normMat.shape: shape of the "+basis_type+" <JKM|JKM> basis: " + str(normMat.shape)+" \n")
	norm_check_write.write("\n")
	norm_check_write.write("\n")

	for s1 in range(njkm):
		for s2 in range(njkm):
			if (np.abs(normMat[s1,s2]) > tol):
				norm_check_write.write("L vec Rotor1: "+str(njkmQuantumNumList[s1,0])+" "+str(njkmQuantumNumList[s1,1])+" "+str(njkmQuantumNumList[s1,2])+"\n")
				norm_check_write.write("R vec Rotor1: "+str(njkmQuantumNumList[s2,0])+" "+str(njkmQuantumNumList[s2,1])+" "+str(njkmQuantumNumList[s2,2])+"\n")
				norm_check_write.write("Norm: "+str(normMat[s1,s2])+"\n")
				norm_check_write.write("\n")
	norm_check_write.close()

def get_pot(size_theta,size_phi,val,xGL,phixiGridPts):

	'''
	Construction of potential matrix begins
	'''

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
	#Eulang2=[0.0, 0.0, 0.0] 
	Eulang2=[0.0, math.pi, 0.0]
	v1d = np.zeros(size_theta*size_phi*size_phi,float)
	ii = 0
	for th1 in range(size_theta):
		for ph1 in range(size_phi):
			for ch1 in range(size_phi):
				ii = ch1+(ph1+th1*size_phi)*size_phi
				Eulang1=[phixiGridPts[ph1], math.acos(xGL[th1]), phixiGridPts[ch1]]
				v1d[ii]=qpot.caleng(com1,com2,Eulang1,Eulang2)

	return v1d

def get_norm(prefile,strFile,basis_type,v1d,eEEebasisuse,Hpot,njkm,njkmQuantumNumList,tol):
	pot_check_file = prefile+"pot-check-"+strFile
	pot_check_write = open(pot_check_file,'w')
	pot_check_write.write("Printing of shapes and elements of potential matrix - "+"\n")
	pot_check_write.write("\n")
	pot_check_write.write("\n")
	pot_check_write.write("shape of potential matrix over three Euler angles : " + str(v1d.shape)+" \n")
	pot_check_write.write("eEEebasisuse.shape: reduced shape of the "+basis_type+" |JKM> basis: " + str(eEEebasisuse.shape)+" \n")
	pot_check_write.write("shape of Hpot : " + str(Hpot.shape)+" \n")
	pot_check_write.write("\n")
	pot_check_write.write("\n")

	for s1 in range(njkm):
		for s2 in range(njkm):
			if (np.abs(Hpot[s1,s2]) > tol):
				pot_check_write.write("L vec Rotor1: "+str(njkmQuantumNumList[s1,0])+" "+str(njkmQuantumNumList[s1,1])+" "+str(njkmQuantumNumList[s1,2])+"\n")
				pot_check_write.write("R vec Rotor1: "+str(njkmQuantumNumList[s2,0])+" "+str(njkmQuantumNumList[s2,1])+" "+str(njkmQuantumNumList[s2,2])+"\n")
				pot_check_write.write("Constant potential field - Re: "+str(np.real(Hpot[s1,s2]))+"   Im: "+str(np.imag(Hpot[s1,s2]))+"\n")
				pot_check_write.write("\n")
	pot_check_write.close()

def get_rot(njkm,njkmQuantumNumList,Ah2o,Bh2o,Ch2o,off_diag):

	'''
	construction of kinetic energy matrix - BEGINS
	'''

	Hrot = np.zeros((njkm,njkm),dtype=float)
    
	for jkm in range(njkm):
		for jkmp in range(njkm):
			if ((njkmQuantumNumList[jkm,0]==njkmQuantumNumList[jkmp,0]) and (njkmQuantumNumList[jkm,2]==njkmQuantumNumList[jkmp,2])):
				if (njkmQuantumNumList[jkm,1]==(njkmQuantumNumList[jkmp,1]-2)):
					Hrot[jkm,jkmp] += 0.25*(Ah2o-Ch2o)*off_diag(njkmQuantumNumList[jkm,0],njkmQuantumNumList[jkm,1])*off_diag(njkmQuantumNumList[jkm,0],njkmQuantumNumList[jkm,1]+1)
				elif (njkmQuantumNumList[jkm,1]==(njkmQuantumNumList[jkmp,1]+2)):
					Hrot[jkm,jkmp] += 0.25*(Ah2o-Ch2o)*off_diag(njkmQuantumNumList[jkm,0],njkmQuantumNumList[jkm,1]-1)*off_diag(njkmQuantumNumList[jkm,0],njkmQuantumNumList[jkm,1]-2)
				elif (njkmQuantumNumList[jkm,1]==(njkmQuantumNumList[jkmp,1])):
					Hrot[jkm,jkmp] += (0.5*(Ah2o + Ch2o)*(njkmQuantumNumList[jkm,0]*(njkmQuantumNumList[jkm,0]+1)) + (Bh2o - 0.5*(Ah2o+Ch2o)) * ((njkmQuantumNumList[jkm,1])**2))

	return Hrot

def wigner_basisre(njkm_J,njkm_K,njkm_M,theta,wt,phi,wp,chi,wc):

	theta0 = math.sqrt((2.*njkm_J+1)/(8.*math.pi**2))*littleD(njkm_J,0,0,np.arccos(theta))*math.sqrt(wt)*math.sqrt(wp)*math.sqrt(wc)
	dd = math.sqrt((2.*njkm_J+1)/(4.*math.pi**2))*littleD(njkm_J,njkm_M,njkm_K,np.arccos(theta))*math.sqrt(wt)
	thetac = dd*math.cos(phi*njkm_M+chi*njkm_K)*math.sqrt(wp)*math.sqrt(wc)
	thetas = dd*math.sin(phi*njkm_M+chi*njkm_K)*math.sqrt(wp)*math.sqrt(wc)

	return theta0,thetac,thetas

def get_basisre(Jmax,njkm,size_theta,size_phi,xGL,wGL,phixiGridPts,dphixi):

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
					theta0,thetac,thetas = wigner_basisre(J,K,M,theta,wt,phi,wp,chi,wc)
					basisf[ib,itpc]=theta0
					ib=ib+1

					for M in range(1,J+1,1):
						theta0,thetac,thetas = wigner_basisre(J,K,M,theta,wt,phi,wp,chi,wc)
						basisf[ib,itpc]=thetac
						ib=ib+1
						basisf[ib,itpc]=thetas
						ib=ib+1
						
					for K in range(1,J+1):
						for M in range(-J,J+1,1):
							theta0,thetac,thetas = wigner_basisre(J,K,M,theta,wt,phi,wp,chi,wc)
							basisf[ib,itpc]=thetac
							ib=ib+1
							basisf[ib,itpc]=thetas
							ib=ib+1
					
	
	return basisf

def get_normbasisre(prefile,strFile,basis_type,eEEebasisuse,normMat,njkm,tol):
	norm_check_file = prefile+"norm-check-"+strFile
	norm_check_write = open(norm_check_file,'w')
	norm_check_write.write("eEEebasisuse.shape: reduced shape of the "+basis_type+" |JKM> basis: " + str(eEEebasisuse.shape)+" \n")
	norm_check_write.write("normMat.shape: shape of the "+basis_type+" <JKM|JKM> basis: " + str(normMat.shape)+" \n")
	norm_check_write.write("\n")
	norm_check_write.write("\n")

	for s1 in range(njkm):
		for s2 in range(njkm):
			if (np.abs(normMat[s1,s2]) > tol):
				norm_check_write.write("L vec Rotor1: "+str(s1)+"\n")
				norm_check_write.write("R vec Rotor1: "+str(s2)+"\n")
				norm_check_write.write("Norm: "+str(normMat[s1,s2])+"\n")
				norm_check_write.write("\n")
	norm_check_write.close()

def get_njkmQuantumNumList(Jmax,njkm):

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


if __name__ == '__main__':    
	zCOM=float(sys.argv[1])
	Jmax=int(sys.argv[2])
	spin_isomer = sys.argv[3]

	size_theta = int(2*Jmax+3)
	size_phi = int(2*(2*Jmax+1))

	tol = 10e-8
	#print the normalization 
	norm_check = True
	io_write = True
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
		
	'''
	JKMQuantumNumList = np.zeros((JKM,3),int)
	JKeMQuantumNumList = np.zeros((JKeM,3),int)
	JKoMQuantumNumList = np.zeros((JKoM,3),int)

	jtempcounter = 0
	for J in range(Jmax+1):
		for K in range(-J,J+1,1):
			for M in range(-J,J+1):
				JKMQuantumNumList[jtempcounter,0]=J
				JKMQuantumNumList[jtempcounter,1]=K
				JKMQuantumNumList[jtempcounter,2]=M
				jtempcounter+=1

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
	'''
	njkm = JKM	
	njkmQuantumNumList = get_njkmQuantumNumList(Jmax,njkm)

	eEEebasisuse = get_basisre(Jmax,njkm,size_theta,size_phi,xGL,wGL,phixiGridPts,dphixi)
	if (norm_check == True):
		normMat = np.tensordot(eEEebasisuse, eEEebasisuse, axes=([1],[1]))
		get_normbasisre(prefile,strFile,basis_type,eEEebasisuse,normMat,njkm,tol)

	v1d = get_pot(size_theta,size_phi,zCOM,xGL,phixiGridPts)

	tempa = v1d[np.newaxis,:]*eEEebasisuse
	Hpot = np.tensordot(eEEebasisuse, tempa, axes=([1],[1]))

	#if (pot_write == True):
	#	get_norm(prefile,strFile,basis_type,v1d,eEEebasisuse,Hpot,njkm,njkmQuantumNumList,tol)

	Hrot = get_rot(njkm,njkmQuantumNumList,Ah2o,Bh2o,Ch2o,off_diag)
    
	Htot = Hrot #+ Hpot

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

