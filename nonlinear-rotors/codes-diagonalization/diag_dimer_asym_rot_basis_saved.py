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
	zCOM=float(sys.argv[1])
	Jmax=int(sys.argv[2])
	incr=int(sys.argv[3])
	#thetaNum = int(2*Jmax+1+incr)
	#angleNum = int(2*Jmax+1+incr)
	thetaNum = 5
	angleNum = 7
	print("Jmax = ", Jmax, flush=True)
	print("angleNum = ", angleNum, flush=True)
	strFile = "diag-2-p-H2O-jmax"+str(Jmax)+"-Rpt"+str(zCOM)+"Angstrom-grid-"+str(thetaNum)+"-"+str(angleNum)+"-saved-basis.txt"
	tol = 10e-8
	
	#print the normalization 
	normCheckMJKeM = False
	normCheckKJKeM = False
	potwrite = False
	io_write = True

	#The rotational A, B, C constants are indicated by Ah2o, Bh2o and Ch2o, respectively. The unit is cm^-1. 
	Ah2o= 27.877 #cm-1 
	Bh2o= 14.512 #cm-1
	Ch2o= 9.285  #cm-1
	CMRECIP2KL = 1.4387672;       	# cm^-1 to Kelvin conversion factor
	Ah2o=Ah2o*CMRECIP2KL
	Bh2o=Bh2o*CMRECIP2KL
	Ch2o=Ch2o*CMRECIP2KL

	xGL,wGL = np.polynomial.legendre.leggauss(thetaNum)              
	phixiGridPts = np.linspace(0,2*np.pi,angleNum, endpoint=False)  
	dphixi = 2.0*np.pi/angleNum                                 
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

	#Construction of potential matrix begins
	com1=[0.0,0.0,0.0]
	com2=[0.0,0.0,zCOM]
	v6d = np.zeros((len(xGL)*angleNum*angleNum,len(xGL)*angleNum*angleNum),float)
	for th1 in range(len(xGL)):
		for ph1 in range(angleNum):
			for ch1 in range(angleNum):
				ii = ch1+(ph1+th1*angleNum)*angleNum
				Eulang1=[phixiGridPts[ph1], math.acos(xGL[th1]), phixiGridPts[ch1]]

				for th2 in range(len(xGL)):
					for ph2 in range(angleNum):
						for ch2 in range(angleNum):
							jj = ch2+(ph2+th2*angleNum)*angleNum
							Eulang2=[phixiGridPts[ph2], math.acos(xGL[th2]), phixiGridPts[ch2]]
							v6d[ii,jj]=pot.caleng(com1,com2,Eulang1,Eulang2)
	#Construction of potential matrix ends

	'''
	for i in range(len(xGL)*angleNum*angleNum):
		print(v6d[0,i])
	exit()
	'''

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
	'''
	theta = 1.0 # in degree
	for s in range(JKeM):
		print("j=",JKeMQuantumNumList[s,0],"m=",JKeMQuantumNumList[s,2],"k=",JKeMQuantumNumList[s,1],littleD(JKeMQuantumNumList[s,0],JKeMQuantumNumList[s,2],JKeMQuantumNumList[s,1],theta*np.pi/180.))
	'''

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

	#block for normalization checking begins
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
	#block for normalization checking ends

	io_write1 = False
	if (io_write1 == True):
		eEEbasisuseTest = KJKeM[:,np.newaxis,np.newaxis,:]*MJKeM[:,np.newaxis,:,np.newaxis]*dJKeM[:,:,np.newaxis,np.newaxis]
		eEEebasisuseTest = np.reshape(eEEbasisuseTest,(JKeM,len(xGL)*angleNum*angleNum),order='C')

		for i in range(JKeM):
			for j in range(len(xGL)*angleNum*angleNum):
				print(" i "+str(i)+" j "+str(j)+" basis "+str(eEEebasisuseTest[i,j]))
		exit()

	#block for construction of <omega1,omega2|J1K1M1,J2K2M2> basis begins 
	eEEbasisuse = KJKeM[:,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]*MJKeM[:,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]*dJKeM[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]*KJKeM[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,:]*MJKeM[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,:,np.newaxis]*dJKeM[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:,np.newaxis,np.newaxis]
	eEEebasisuse = np.reshape(eEEbasisuse,(JKeM,len(xGL)*angleNum*angleNum,JKeM,len(xGL)*angleNum*angleNum),order='C')
	normMat = np.tensordot(eEEebasisuse, np.conjugate(eEEebasisuse), axes=([1,3],[1,3]))
	#block for construction of |J1K1M1,J2K2M2> basis ends

	#Norms are Saved in norm-check.dat
	#printing block is opened
	norm_check_file = "norm-check"+strFile
	norm_check_write = open(norm_check_file,'w')
	norm_check_write.write("eEEbasisuse.shape: shape of the even |J1K1M1>|J2K2M2> basis: " + str(eEEbasisuse.shape)+" \n")
	norm_check_write.write("eEEebasisuse.shape: reduced shape of the even |J1K1M1>|J2K2M2> basis: " + str(eEEebasisuse.shape)+" \n")
	norm_check_write.write("normMat.shape: shape of the even <J1K1M1|<J2K2M2||J2K2M2>|J1K1M1> basis: " + str(normMat.shape)+" \n")
	norm_check_write.write("\n")
	norm_check_write.write("\n")

	for s1 in range(JKeM):
		for s2 in range(JKeM):
			for s3 in range(JKeM):
				for s4 in range(JKeM):
					if (np.abs(normMat[s1,s2,s3,s4]) > tol):
						norm_check_write.write("L vec Rotor1: "+str(JKeMQuantumNumList[s1,0])+" "+str(JKeMQuantumNumList[s1,1])+" "+str(JKeMQuantumNumList[s1,2])+"\n")
						norm_check_write.write("R vec Rotor1: "+str(JKeMQuantumNumList[s3,0])+" "+str(JKeMQuantumNumList[s3,1])+" "+str(JKeMQuantumNumList[s3,2])+"\n")
						norm_check_write.write("L vec Rotor2: "+str(JKeMQuantumNumList[s2,0])+" "+str(JKeMQuantumNumList[s2,1])+" "+str(JKeMQuantumNumList[s2,2])+"\n")
						norm_check_write.write("R vec Rotor2: "+str(JKeMQuantumNumList[s4,0])+" "+str(JKeMQuantumNumList[s4,1])+" "+str(JKeMQuantumNumList[s4,2])+"\n")
						norm_check_write.write("Norm: Real "+str(np.real(normMat[s1,s2,s3,s4]))+"\n")
						norm_check_write.write("Norm: Imag "+str(np.imag(normMat[s1,s2,s3,s4]))+"\n")
						norm_check_write.write("\n")
	norm_check_write.close()
	#printing block is closed

	#Construction of a constant potential matrix over the six Euler angles
	#v6d = np.zeros((len(xGL)*angleNum*angleNum, len(xGL)*angleNum*angleNum),dtype=float)
	#v6d.fill(100.)
	v6d = v6d[np.newaxis,:,np.newaxis,:]
	tempa = v6d*eEEebasisuse
	HpotKee = np.tensordot(np.conjugate(eEEebasisuse), tempa, axes=([1,3],[1,3]))

	#printing block is opened
	if (io_write == True):
		pot_check_file = "pot-check"+strFile
		pot_check_write = open(pot_check_file,'w')
		pot_check_write.write("Printing of shapes and elements of potential matrix - "+"\n")
		pot_check_write.write("\n")
		pot_check_write.write("\n")
		pot_check_write.write("shape of potential matrix over six Euler angles : " + str(v6d.shape)+" \n")
		pot_check_write.write("eEEebasisuse.shape: reduced shape of the even |J1K1M1>|J2K2M2> basis: " + str(eEEebasisuse.shape)+" \n")
		pot_check_write.write("shape of Hpot : " + str(HpotKee.shape)+" \n")
		pot_check_write.write("\n")
		pot_check_write.write("\n")

		for s1 in range(JKeM):
			for s2 in range(JKeM):
				for s3 in range(JKeM):
					for s4 in range(JKeM):
						if (np.abs(HpotKee[s1,s2,s3,s4]) > 10e-2):
							pot_check_write.write("L vec Rotor1: s1 "+str(s1)+"  "+str(JKeMQuantumNumList[s1,0])+" "+str(JKeMQuantumNumList[s1,1])+" "+str(JKeMQuantumNumList[s1,2])+"\n")
							pot_check_write.write("R vec Rotor1: s3 "+str(s3)+"  "+str(JKeMQuantumNumList[s3,0])+" "+str(JKeMQuantumNumList[s3,1])+" "+str(JKeMQuantumNumList[s3,2])+"\n")
							pot_check_write.write("L vec Rotor2: s2 "+str(s2)+"  "+str(JKeMQuantumNumList[s2,0])+" "+str(JKeMQuantumNumList[s2,1])+" "+str(JKeMQuantumNumList[s2,2])+"\n")
							pot_check_write.write("R vec Rotor2: s4 "+str(s4)+"  "+str(JKeMQuantumNumList[s4,0])+" "+str(JKeMQuantumNumList[s4,1])+" "+str(JKeMQuantumNumList[s4,2])+"\n")
							pot_check_write.write("Potential: Real "+str(np.real(HpotKee[s1,s2,s3,s4]))+"\n")
							pot_check_write.write("Potential: Imag "+str(np.imag(HpotKee[s1,s2,s3,s4]))+"\n")
							pot_check_write.write("\n")
		pot_check_write.close()
	'''
	s1=1
	s2=0
	s3=1
	s4=0
	print("L vec Rotor1: s1 "+str(s1)+"  "+str(JKeMQuantumNumList[s1,0])+" "+str(JKeMQuantumNumList[s1,1])+" "+str(JKeMQuantumNumList[s1,2]))
	print("L vec Rotor2: s2 "+str(s2)+"  "+str(JKeMQuantumNumList[s2,0])+" "+str(JKeMQuantumNumList[s2,1])+" "+str(JKeMQuantumNumList[s2,2]))
	print("R vec Rotor1: s3 "+str(s3)+"  "+str(JKeMQuantumNumList[s3,0])+" "+str(JKeMQuantumNumList[s3,1])+" "+str(JKeMQuantumNumList[s3,2]))
	print("R vec Rotor2: s4 "+str(s4)+"  "+str(JKeMQuantumNumList[s4,0])+" "+str(JKeMQuantumNumList[s4,1])+" "+str(JKeMQuantumNumList[s4,2]))
	print(HpotKee[s1,s2,s3,s4])
	exit()
	'''
	# printing block is closed

    #Computation of Hrot (Asymmetric Top Hamiltonian in Symmetric Top Basis)
	HpotKee = np.reshape(HpotKee,(JKeeM,JKeeM),order='C')
	HrotKee = np.zeros((JKeeM,JKeeM),dtype=float)

	#print(np.any(HpotKee.imag > 10e-13))
	#print(np.any(HpotKee.real > 10e-13))

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

	HtotKee = HrotKee + HpotKee   #Unit Kelvin

	if (np.allclose(HtotKee, HtotKee.T) == False):
		print("The Hamiltonian matrx - 'HtotKe' is not hermitian.")
		exit()
		
# printing block is open
	"""
	tot_est = LA.eigh(Htot)[0] 
	sort_indx = tot_est.argsort()     # prints out eigenvalues for pure asymmetric top rotor (z_ORTHOz)
	tot_est = tot_est[sort_indx]       

	#printing block is opened
	tot_est_comb = np.array([tot_est, tot_est/CMRECIP2KL])

	eig_file = "eigen-values"+strFile
	np.savetxt("exact-energies-of-H2O/"+eig_file, tot_est_comb.T, fmt='%20.8f', delimiter=' ', header='Eigen values of (Htot = Hrot + Hvpot) - Units associated with the first and second columns are Kelvin and wavenumber, respectively. ')
	"""
	# printing block is closed
	evals_large, evecs_large = eigsh(HtotKee, 5, which='SA')
	#printing block is opened
	tot_est_comb = np.array([evals_large, evals_large/CMRECIP2KL])

	eig_file = "boundstates-"+strFile
	np.savetxt(eig_file, tot_est_comb.T, fmt='%20.8f', delimiter=' ', header='Eigen values of (Htot = Hrot + Hvpot) - Units associated with the first and second columns are Kelvin and wavenumber, respectively. ')
	# printing block is closed
