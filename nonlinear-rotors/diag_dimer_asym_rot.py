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
	Jmax=int(sys.argv[1])
	angleNum = int(sys.argv[2])
	print("Jmax = ", Jmax, flush=True)
	print("angleNum = ", angleNum, flush=True)
	strFile = "-Jmax-"+str(Jmax)+"-grid-"+str(angleNum)+".txt"
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

	JKeeM=int(JKeM*JKeM)
	JKeoM=int(JKeM*JKoM)
	JKoeM=int(JKoM*JKeM)
	JKooM=int(JKoM*JKoM)

	ChkJKM = JKeeM+JKeoM+JKoeM+JKooM
	JKM2 = int(JKM*JKM)

	if (ChkJKM != JKM2):
		print("Wrong index estimation ...")
		exit()
    
	print("|------------------------------------------------")
	print("| Number of basis functions calculations ....")
	print("| ")
	print("| # of |JKM> basis = "+str(JKM))
	print("| ")
	print("| # of even K in |JKM> = "+str(JKeM))
	print("| # of odd  K in |JKM> = "+str(JKoM))
	print("| ")
	print("| # of even K1, even K2 in the |J1K1M1,J2K2M2> = "+str(JKeeM))
	print("| # of even K1, odd  K2 in the |J1K1M1,J2K2M2> = "+str(JKeoM))
	print("| # of odd  K1, even K2 in the |J1K1M1,J2K2M2> = "+str(JKoeM))
	print("| # of odd  K1, odd  K2 in the |J1K1M1,J2K2M2> = "+str(JKooM))
	print("| ")
	print("| # of |J1K1M1;J2K2M2> basis= # of ChkJKM")
	print("| # of |J1K1M1;J2K2M2> basis= "+str(JKM2))
	print("| # of ChkJKM = " + str(ChkJKM))
	print("|------------------------------------------------")
	sys.stdout.flush()
    
	JKeMQuantumNumList = np.zeros((JKeM,3),int)
	JKoMQuantumNumList = np.zeros((JKoM,3),int)

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
    #odd
	jtempcounter = 0
	for J in range(Jmax+1):
		if J%2==0:
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
	Nk = 1.
	for s in range(JKeM):
		for th in range(len(xGL)):
			dJKeM[s,th] = np.sqrt((2.*JKeMQuantumNumList[s,0]+1)/(8.*np.pi**2))*littleD(JKeMQuantumNumList[s,0],JKeMQuantumNumList[s,2],JKeMQuantumNumList[s,1],np.arccos(xGL[th]))*np.sqrt(wGL[th])

		for ph in range(angleNum):
			KJKeM[s,ph] = np.exp(1j*phixiGridPts[ph]*JKeMQuantumNumList[s,1])*np.sqrt(dphixi)*Nk
			MJKeM[s,ph] = np.exp(1j*phixiGridPts[ph]*JKeMQuantumNumList[s,2])*np.sqrt(dphixi)*Nk
	#block for construction of individual basis ends

    #Computation of Hrot (Asymmetric Top Hamiltonian in Symmetric Top Basis)
	HrotKee = np.zeros((JKeeM,JKeeM),dtype=float)

	jkm12 = 0 
	for jkm1 in range(JKeM):
		for jkm2 in range(JKeM):
			jkmp12 = 0 
			for jkmp1 in range(JKeM):
				for jkmp2 in range(JKeM):
					#For 1st rotor
					if ((JKeMQuantumNumList[jkm2,0]==JKeMQuantumNumList[jkmp2,0]) and (JKeMQuantumNumList[jkm2,1]==JKeMQuantumNumList[jkmp2,1]) and (JKeMQuantumNumList[jkm2,2]==JKeMQuantumNumList[jkmp2,2])):
						if ((JKeMQuantumNumList[jkm1,0]==JKeMQuantumNumList[jkmp1,0]) and (JKeMQuantumNumList[jkm1,2]==JKeMQuantumNumList[jkmp1,2])):
							if (JKeMQuantumNumList[jkm1,1]==(JKeMQuantumNumList[jkmp1,1]-2)):
								HrotKee[jkm12,jkmp12] += 0.25*(Ah2o-Ch2o)*off_diag(JKeMQuantumNumList[jkm1,0],JKeMQuantumNumList[jkm1,1])*off_diag(JKeMQuantumNumList[jkm1,0],JKeMQuantumNumList[jkm1,1]+1)
							elif (JKeMQuantumNumList[jkm1,1]==(JKeMQuantumNumList[jkmp1,1]+2)):
								HrotKee[jkm12,jkmp12] += 0.25*(Ah2o-Ch2o)*off_diag(JKeMQuantumNumList[jkm1,0],JKeMQuantumNumList[jkm1,1]-1)*off_diag(JKeMQuantumNumList[jkm1,0],JKeMQuantumNumList[jkm1,1]-2)
							elif (JKeMQuantumNumList[jkm1,1]==(JKeMQuantumNumList[jkmp1,1])):
								HrotKee[jkm12,jkmp12] += (0.5*(Ah2o + Ch2o)*(JKeMQuantumNumList[jkm1,0]*(JKeMQuantumNumList[jkm1,0]+1)) + (Bh2o - 0.5*(Ah2o+Ch2o)) * ((JKeMQuantumNumList[jkm1,1])**2))
					#For 2nd rotor
					if ((JKeMQuantumNumList[jkm1,0]==JKeMQuantumNumList[jkmp1,0]) and (JKeMQuantumNumList[jkm1,1]==JKeMQuantumNumList[jkmp1,1]) and (JKeMQuantumNumList[jkm1,2]==JKeMQuantumNumList[jkmp1,2])):
						if ((JKeMQuantumNumList[jkm2,0]==JKeMQuantumNumList[jkmp2,0]) and (JKeMQuantumNumList[jkm2,2]==JKeMQuantumNumList[jkmp2,2])):
							if (JKeMQuantumNumList[jkm2,1]==(JKeMQuantumNumList[jkmp2,1]-2)):
								HrotKee[jkm12,jkmp12] += 0.25*(Ah2o-Ch2o)*off_diag(JKeMQuantumNumList[jkm2,0],JKeMQuantumNumList[jkm2,1])*off_diag(JKeMQuantumNumList[jkm2,0],JKeMQuantumNumList[jkm2,1]+1)
							elif (JKeMQuantumNumList[jkm2,1]==(JKeMQuantumNumList[jkmp2,1]+2)):
								HrotKee[jkm12,jkmp12] += 0.25*(Ah2o-Ch2o)*off_diag(JKeMQuantumNumList[jkm2,0],JKeMQuantumNumList[jkm2,1]-1)*off_diag(JKeMQuantumNumList[jkm2,0],JKeMQuantumNumList[jkm2,1]-2)
							elif (JKeMQuantumNumList[jkm2,1]==(JKeMQuantumNumList[jkmp2,1])):
								HrotKee[jkm12,jkmp12] += (0.5*(Ah2o + Ch2o)*(JKeMQuantumNumList[jkm2,0]*(JKeMQuantumNumList[jkm2,0]+1)) + (Bh2o - 0.5*(Ah2o+Ch2o)) * ((JKeMQuantumNumList[jkm2,1])**2))
					jkmp12 += 1
			jkm12 += 1

    #Computation of Hpot (Asymmetric Top Hamiltonian in Symmetric Top Basis)
	HpotKee = np.zeros((JKeeM,JKeeM),dtype=complex)
	normMat = np.zeros((JKeeM,JKeeM),dtype=complex)

	com1=[0.0,0.0,0.0]
	com2=[0.0,0.0,10.05]
	for th1 in range(len(xGL)):
		for ph1 in range(angleNum):
			for ch1 in range(angleNum):
				Eulang1=[math.acos(xGL[th1]),phixiGridPts[ph1],phixiGridPts[ch1]]

				for th2 in range(len(xGL)):
					for ph2 in range(angleNum):
						for ch2 in range(angleNum):
							Eulang2=[math.acos(xGL[th2]),phixiGridPts[ph2],phixiGridPts[ch2]]
							v6d=20.0#pot.caleng(com1,com2,Eulang1,Eulang2)

							jkm12 = 0 
							for jkm1 in range(JKeM):
								lvecKee = dJKeM[jkm1,th1]*MJKeM[jkm1,ph1]*KJKeM[jkm1,ch1]
								for jkm2 in range(JKeM):
									lvecKee *= dJKeM[jkm2,th2]*MJKeM[jkm2,ph2]*KJKeM[jkm2,ch2]

									jkmp12 = 0 
									for jkmp1 in range(JKeM):
										rvecKee = dJKeM[jkmp1,th1]*MJKeM[jkmp1,ph1]*KJKeM[jkmp1,ch1]
										for jkmp2 in range(JKeM):
											rvecKee *= dJKeM[jkmp2,th2]*MJKeM[jkmp2,ph2]*KJKeM[jkmp2,ch2]
											
											normMat[jkm12,jkmp12] += np.conjugate(lvecKee)*rvecKee
											HpotKee[jkm12,jkmp12] += np.conjugate(lvecKee)*v6d*rvecKee
											jkmp12 += 1
									jkm12 += 1


	#Norms are Saved in norm-check.dat
	#printing block is opened
	norm_check_file = "norm-check"+strFile
	norm_check_write = open(norm_check_file,'w')
	ii=0
	for s1 in range(JKeM):
		for s2 in range(JKeM):
			jj=0
			for s3 in range(JKeM):
				for s4 in range(JKeM):
					if (np.abs(normMat[ii,jj]) > tol):
						norm_check_write.write("L vec Rotor1: "+str(JKeMQuantumNumList[s1,0])+" "+str(JKeMQuantumNumList[s1,1])+" "+str(JKeMQuantumNumList[s1,2])+"\n")
						norm_check_write.write("R vec Rotor1: "+str(JKeMQuantumNumList[s3,0])+" "+str(JKeMQuantumNumList[s3,1])+" "+str(JKeMQuantumNumList[s3,2])+"\n")
						norm_check_write.write("L vec Rotor2: "+str(JKeMQuantumNumList[s2,0])+" "+str(JKeMQuantumNumList[s2,1])+" "+str(JKeMQuantumNumList[s2,2])+"\n")
						norm_check_write.write("R vec Rotor2: "+str(JKeMQuantumNumList[s4,0])+" "+str(JKeMQuantumNumList[s4,1])+" "+str(JKeMQuantumNumList[s4,2])+"\n")
						norm_check_write.write("Norm: "+str(normMat[ii,jj])+"\n")
						norm_check_write.write("\n")
					jj=jj+1
			ii=ii+1
	norm_check_write.close()
	sys.stdout.flush()
	#printing block is closed

	HtotKee = HrotKee + HpotKee   #Unit Kelvin

	# check to make sure H is hermitian
	if (np.all(np.abs(HtotKee-HtotKee.T) < tol) == False):
		print("The Hamiltonian matrx HtotKe is not hermitian.")
		exit()

	evals_large, evecs_large = eigsh(HtotKee, 3, which='SA')

	#printing block is opened
	tot_est_comb = np.array([evals_large, evals_large/CMRECIP2KL])

	eig_file = "eigen-values"+strFile
	np.savetxt(eig_file, tot_est_comb.T, fmt='%20.8f', delimiter=' ', header='Eigen values of (HtotKee = Hrot + Hvpot) - Units associated with the first and second columns are Kelvin and wavenumber, respectively. ')
	# printing block is closed
