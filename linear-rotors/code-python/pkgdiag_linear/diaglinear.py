#Final version of diagonalizing H2O-H2O
#
#  Features:
#   - compute the eigenvalues and wavefunctions for the full 6D problen
#	- consider only even K 
#
import sys
import math
import numpy as np
from scipy import special as sp
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

def wigner_basis(njkm,size_theta,size_phi,njkmQuantumNumList,xGL,wGL,phixiGridPts,dphixi):

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
	#Eulang2=[0.0,0.0,0.0] 
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

def get_rotmat(njkm,njkmQuantumNumList,Ah2o,Bh2o,Ch2o):

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

def get_numbbasis(njkm,Jmax,spin_isomer):

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

def normalization_check(prefile,strFile,basis_type,eEEbasisuse,eEEebasisuse,normMat,njkm,njkmQuantumNumList,tol):
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
				norm_check_write.write("Constant potential field - Re: "+str(np.real(normMat[s1,s2]))+"   Im: "+str(np.imag(normMat[s1,s2]))+"\n")
				norm_check_write.write("\n")
	norm_check_write.close()

def get_numbbasisLinear(njm,Jmax,spin_isomer):

	if (spin_isomer == "spinless"):
		JM=njm
		JMQuantumNumList = np.zeros((JM,2),int)
		#all J
		jtempcounter = 0
		for J in range(Jmax+1):
			for M in range(-J,J+1):
				JMQuantumNumList[jtempcounter,0]=J
				JMQuantumNumList[jtempcounter,1]=M
				jtempcounter+=1
		return JMQuantumNumList

	if (spin_isomer == "para"):
		JeM=njm
		JeMQuantumNumList = np.zeros((JeM,2),int)
		#even
		jtempcounter = 0
		for J in range(0,Jmax+1,2):
			for M in range(-J,J+1):
				JeMQuantumNumList[jtempcounter,0]=J
				JeMQuantumNumList[jtempcounter,1]=M
				jtempcounter+=1
		return JeMQuantumNumList

	if (spin_isomer == "ortho"):
		JoM=njm
		JoMQuantumNumList = np.zeros((JoM,2),int)
		#odd
		jtempcounter = 0
		for J in range(1,Jmax+1,2):
			for M in range(-J,J+1):
				JoMQuantumNumList[jtempcounter,0]=J
				JoMQuantumNumList[jtempcounter,1]=M
				jtempcounter+=1

		return JoMQuantumNumList

def normalization_checkLinear(prefile,strFile,basis_type,eEEbasisuse,normMat,njm,njmQuantumNumList,tol):
	norm_check_file = prefile+"norm-check-"+strFile
	norm_check_write = open(norm_check_file,'w')
	norm_check_write.write("eEEbasisuse.shape: shape of the "+basis_type+" |JM> basis: " + str(eEEbasisuse.shape)+" \n")
	norm_check_write.write("normMat.shape: shape of the "+basis_type+" <JM|JM> basis: " + str(normMat.shape)+" \n")
	norm_check_write.write("\n")
	norm_check_write.write("\n")

	for s1 in range(njm):
		for s2 in range(njm):
			if (np.abs(normMat[s1,s2]) > tol):
				norm_check_write.write("L vec Rotor1: "+str(njmQuantumNumList[s1,0])+" "+str(njmQuantumNumList[s1,1])+"\n")
				norm_check_write.write("R vec Rotor1: "+str(njmQuantumNumList[s2,0])+" "+str(njmQuantumNumList[s2,1])+"\n")
				norm_check_write.write("Constant potential field - Re: "+str(np.real(normMat[s1,s2]))+"   Im: "+str(np.imag(normMat[s1,s2]))+"\n")
				norm_check_write.write("\n")
	norm_check_write.close()

def spherical_harmonicsReal(njm,size_theta,size_phi,njmQuantumNumList,xGL,wGL,phixiGridPts,dphixi):

	'''
	construnction of real spherical harmonics
	'''
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
	basisfun=np.zeros((size_theta*size_phi,njm),complex)
	for jm in range(njm):
		for th in range(size_theta):
			for ph in range(size_phi):
				ii = ph+th*size_phi
				basisfun[ii,jm]=sp.sph_harm(njmQuantumNumList[jm,1],njmQuantumNumList[jm,0],phixiGridPts[ph],np.arccos(xGL[th]))*np.sqrt(wGL[th])*np.sqrt(dphixi)
	return basisfun

