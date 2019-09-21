#Final version of diagonalizing H2O-H2O
#
#  Features:
#   - compute the eigenvalues and wavefunctions for the full 6D problen
#   - Symmetry adapted basis:
#         -- Ortho/Para (K-even or K-odd)
import gc
import sys
import os
import numpy as np
import math
#from numpy import linalg as LA
from scipy import linalg as LA
from datetime import datetime
from pprint import pprint
import multiprocessing as mp
#np.show_config()

startTime = datetime.now()

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
	print("Jmax = ", Jmax)
	print("angleNum = ", angleNum)
	
	#print the normalization 
	normCheckMJKeM = False
	normCheckKJKeM = False

	"""
	# Universal constants
	#NA = 6.022140857*(10**23)
	NA = 6.02214076*(10**23)
	h = 6.62607015*(10**-34)
	hbar = h/(2.*np.pi)
	clight = 299792458.
	jpcm = h*clight*100.
	#jpcm = 1.9863*(10**-23)

    #JOULES TO 1/cm  ==>  E = hc/lam   E / hc 
    
    #Molecular constants are given below -
	massh2o = (18.01056/(NA*1000.))

	mO = 15.9994
	mH = 1.008
	mH2O = 2.*mH + mO
	"""
					
	#The rotational A, B, C constants are indicated by Ah2o, Bh2o and Ch2o, respectively. The unit is cm^-1. 
	Ah2o= 27.877 #cm-1 
	Bh2o= 14.512 #cm-1
	Ch2o= 9.285  #cm-1

	"""
	Ost = np.array([0., 0., -0.006563807])
	H1st = np.array([0.07575, 0., 0.052086193])
	H2st = np.array([-0.07575, 0., 0.052086193])

	H2OCoM = (mH*(H1st+H2st) + mO*Ost)/mH2O

	print("H2O CoM: ")
	print(H2OCoM)
	"""

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

	JKM = int(((2*Jmax+1)*(2*Jmax+2)*(2*Jmax+3)/6)) #JKM = "Sum[(2J+1)**4,{J,0,Jmax}]" is computed in mathematica

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
    
	JKeMQuantumNumList = np.zeros((JKeM,3),int)
	JKoMQuantumNumList = np.zeros((JKoM,3),int)

	JKeMreverse={}
	JKoMreverse={}

	#even
	jtempcounter = 0
	for J in range(Jmax+1):
		if J%2==0:
			for K in range(-J,J+1,2):
				for M in range(-J,J+1):
					JKeMQuantumNumList[jtempcounter,0]=J
					JKeMQuantumNumList[jtempcounter,1]=K
					JKeMQuantumNumList[jtempcounter,2]=M
					JKeMreverse[(J,K,M)]=jtempcounter
					jtempcounter+=1
		else:
			for K in range(-J+1,J,2):
				for M in range(-J,J+1):
					JKeMQuantumNumList[jtempcounter,0]=J
					JKeMQuantumNumList[jtempcounter,1]=K
					JKeMQuantumNumList[jtempcounter,2]=M
					JKeMreverse[(J,K,M)]=jtempcounter
					jtempcounter+=1
	#print(jtempcounter)
    #odd
	jtempcounter = 0
	for J in range(Jmax+1):
		if J%2==0:
			for K in range(-J+1,J,2):
				#print(K)
				for M in range(-J,J+1):
					JKoMQuantumNumList[jtempcounter,0]=J
					JKoMQuantumNumList[jtempcounter,1]=K
					JKoMQuantumNumList[jtempcounter,2]=M
					JKoMreverse[(J,K,M)]=jtempcounter
					jtempcounter+=1
		else:
			for K in range(-J,J+1,2):
				#print(K)
				for M in range(-J,J+1):
					JKoMQuantumNumList[jtempcounter,0]=J
					JKoMQuantumNumList[jtempcounter,1]=K
					JKoMQuantumNumList[jtempcounter,2]=M
					JKoMreverse[(J,K,M)]=jtempcounter
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

	KJKeMc = np.zeros((JKeM,angleNum),float)
	KJKeMs = np.zeros((JKeM,angleNum),float)
	MJKeMc = np.zeros((JKeM,angleNum),float)
	MJKeMs = np.zeros((JKeM,angleNum),float)

	Nk = 1./np.sqrt(2.*np.pi)

	for s in range(JKeM):
		"""
		if JKeMQuantumNumList[s,1] == 0:
			Nk = 0.5
		else:
			Nk = 1./np.sqrt(2.)
		"""
		for th in range(len(xGL)):

			dJKeM[s,th] = np.sqrt((2.*JKeMQuantumNumList[s,0]+1)/(8.*np.pi**2))*littleD(JKeMQuantumNumList[s,0],JKeMQuantumNumList[s,2],JKeMQuantumNumList[s,1],np.arccos(xGL[th]))*np.sqrt(wGL[th])

		for ph in range(angleNum):
			KJKeMc[s,ph] = np.cos(phixiGridPts[ph]*JKeMQuantumNumList[s,1])*np.sqrt(dphixi)*Nk
			KJKeMs[s,ph] = np.sin(phixiGridPts[ph]*JKeMQuantumNumList[s,1])*np.sqrt(dphixi)*Nk
			MJKeMc[s,ph] = np.cos(phixiGridPts[ph]*JKeMQuantumNumList[s,2])*np.sqrt(dphixi)*Nk
			MJKeMs[s,ph] = np.sin(phixiGridPts[ph]*JKeMQuantumNumList[s,2])*np.sqrt(dphixi)*Nk

			KJKeM[s,ph] = np.exp(-1j*phixiGridPts[ph]*JKeMQuantumNumList[s,1])*np.sqrt(dphixi)*Nk
			MJKeM[s,ph] = np.exp(-1j*phixiGridPts[ph]*JKeMQuantumNumList[s,2])*np.sqrt(dphixi)*Nk

	#Normalization checking
	if (normCheckMJKeM == True):
		print("Normalization test for |MJKeM> basis ")
		for s1 in range(JKeM):
			for s2 in range(JKeM):
				if (JKeMQuantumNumList[s1,2] != JKeMQuantumNumList[s2,2]):
					print("M1 = ",JKeMQuantumNumList[s1,2]," M2 = ",JKeMQuantumNumList[s2,2], " <M1|M2> = ",np.inner(MJKeM[s1,:],np.conjugate(MJKeM[s2,:])))

		print("")

	if (normCheckKJKeM == True):
		print("Normalization test for |KJKeM> basis ")
		for s1 in range(JKeM):
			for s2 in range(JKeM):
				if (JKeMQuantumNumList[s1,1] != JKeMQuantumNumList[s2,1]):
					print("M1 = ",JKeMQuantumNumList[s1,1]," M2 = ",JKeMQuantumNumList[s2,1], " <M1|M2> = ",np.inner(KJKeM[s1,:],np.conjugate(KJKeM[s2,:])))
    

	eEEbasisuse = KJKeM[:,np.newaxis,np.newaxis,np.newaxis,:]*np.conj(KJKeM[np.newaxis,:,np.newaxis,np.newaxis,:]) * MJKeM[:,np.newaxis,np.newaxis,:,np.newaxis] * np.conj(MJKeM[np.newaxis,:,np.newaxis,:,np.newaxis]) * dJKeM[:,np.newaxis,:,np.newaxis,np.newaxis] * dJKeM[np.newaxis,:,:,np.newaxis,np.newaxis]

	#eIIbasisuse = invKJKeM[:,np.newaxis,np.newaxis,np.newaxis,:]*np.conj(invKJKeM[np.newaxis,:,np.newaxis,np.newaxis,:]) * MJKeM[:,np.newaxis,np.newaxis,:,np.newaxis] * np.conj(MJKeM[np.newaxis,:,np.newaxis,:,np.newaxis]) * invdJKeM[:,np.newaxis,:,np.newaxis,np.newaxis] * invdJKeM[np.newaxis,:,:,np.newaxis,np.newaxis]

	#eEIbasisuse = KJKeM[:,np.newaxis,np.newaxis,np.newaxis,:]*np.conj(invKJKeM[np.newaxis,:,np.newaxis,np.newaxis,:]) * MJKeM[:,np.newaxis,np.newaxis,:,np.newaxis] * np.conj(MJKeM[np.newaxis,:,np.newaxis,:,np.newaxis]) * dJKeM[:,np.newaxis,:,np.newaxis,np.newaxis] * invdJKeM[np.newaxis,:,:,np.newaxis,np.newaxis]

	#eIEbasisuse = invKJKeM[:,np.newaxis,np.newaxis,np.newaxis,:]*np.conj(KJKeM[np.newaxis,:,np.newaxis,np.newaxis,:]) * MJKeM[:,np.newaxis,np.newaxis,:,np.newaxis] * np.conj(MJKeM[np.newaxis,:,np.newaxis,:,np.newaxis]) * invdJKeM[:,np.newaxis,:,np.newaxis,np.newaxis] * dJKeM[np.newaxis,:,:,np.newaxis,np.newaxis]

	print(eEEbasisuse.shape)
	#print("V SIZE: ", v6d.shape)
	#print("PSInlm: ", PSInlm.shape)

	#eEEebasisuse = np.reshape(eEEbasisuse,(JKeM*JKeM,angleNum*angleNum*thetaNum),order='C')
	#eIIebasisuse = np.reshape(eIIbasisuse,(JKeM*JKeM,angleNum*angleNum*thetaNum),order='C')
	#eEIebasisuse = np.reshape(eEIbasisuse,(JKeM*JKeM,angleNum*angleNum*thetaNum),order='C')
	#eIEebasisuse = np.reshape(eIEbasisuse,(JKeM*JKeM,angleNum*angleNum*thetaNum),order='C')
	exit()

	"""
    #Calculate the V element for Au block
	#JKeMrange = range(JKeM)                         
	#JKoMrange = range(JKoM)
	v6d = np.reshape(v6d, (cageGridNum*cageAngleNum*cageAngleNum, thetaNum*angleNum*angleNum), order='C')  #


	print("eEE i")
	tempa = np.tensordot(eEEebasisuse, v6d, axes=([1],[1]))
	tempa = np.reshape(tempa,(JKeM,JKeM,cageGridNum*cageAngleNum*cageAngleNum),order='C')
	vNJKeMEE = np.tensordot(tempa,(PSInlm[:,np.newaxis,:]*PSIconjnlm[np.newaxis,:,:]), axes=([2],[2]))


	sys.stdout.flush()
	print("eII i")
	tempa = np.tensordot(eIIebasisuse, v6d, axes=([1],[1]))
	tempa = np.reshape(tempa,(JKeM,JKeM,cageGridNum*cageAngleNum*cageAngleNum),order='C')
	vNJKeMII = np.tensordot(tempa, (PSInlminv[:,np.newaxis,:]*PSIconjnlminv[np.newaxis,:,:]), axes=([2],[2]))

	sys.stdout.flush()

	print("eEI i")
	tempa = np.tensordot(eEIebasisuse, v6d, axes=([1],[1]))
	tempa = np.reshape(tempa,(JKeM,JKeM,cageGridNum*cageAngleNum*cageAngleNum),order='C')
	vNJKeMEI = np.tensordot(tempa, (PSInlm[:,np.newaxis,:]*PSIconjnlminv[np.newaxis,:,:]), axes=([2],[2]))

	sys.stdout.flush()
	print("eIE i")
	tempa = np.tensordot(eIEebasisuse, v6d, axes=([1],[1]))
	tempa = np.reshape(tempa,(JKeM,JKeM,cageGridNum*cageAngleNum*cageAngleNum),order='C')
	vNJKeMIE = np.tensordot(tempa, (PSInlminv[:,np.newaxis,:]*PSIconjnlm[np.newaxis,:,:]), axes=([2],[2]))

    #Computation of Hrot (Asymmetric Top Hamiltonian in Symmetric Top Basis)
	HrotKee = np.zeros((JKeeM,JKeeM),dtype=float)
	HrotKoo = np.zeros((JKooM,JKooM),dtype=float)

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

	rotest = LA.eigh(HrotKee)[0] 
	azdx = rotest.argsort()     # prints out eigenvalues for pure asymmetric top rotor (z_ORTHOz)
	rotest = rotest[azdx]       
	print(rotest)               
	"""
