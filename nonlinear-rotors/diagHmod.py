#Final version of diagonalizing H2O@C60
#
#  Features:
#   - compute the eigenvalues and wavefunctions for the full 6D problen
#   - Symmetry adapted basis:
#         -- Ortho/Para (K-even or K-odd)
#         -- inversion symmetry epsilon = 0 (symmetric) or 1 (anti-symmetric)
#                            
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
np.show_config()


startTime = datetime.now()

#F U N C T I O N S

####################################################################################################################################
#calculate binomial coefficient
####################################################################################################################################
def binom(n,k):
    minus = n-k
    if minus < 0:
        return 0
    else:
        return (np.math.factorial(n)/ (np.math.factorial(k) * np.math.factorial(minus)))
####################################################################################################################################

###############################
# off diagonal <JKM|H|J'K'M'> #
############################################
def off_diag (j,k):                        
    f = np.sqrt((j*(j+1)) - (k*(k+1)))     
    return f                               
############################################



####################################################################################################################################
# CALCULATE d(m',m, theta) ie. little d-rotation matrix 
####################################################################################################################################
def littleD(ldJ,ldmp,ldm,ldtheta):
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
####################################################################################################################################






##########################################################################################
#  FUNCTION: calcvr(r) :  calculate the potential (THETA,PHI,theta, phi, xi) at a given r    
##########################################################################################
def calcvr(r):
    temp = np.array([[np.zeros(rotOx.shape) for TH in range(cageAngleNum)] for PHI in range(cageAngleNum)])
    counter = 0
    for TH in range(cageAngleNum):
        for PHI in range(len(cagephiGrid)):
            counter+=1
            if counter%100 ==0:
                print(r, counter)
            v = 0.
            x = cagerGrid[r] * np.sin(np.arccos(cagexGL[TH])) * np.cos(cagephiGrid[PHI]) * (10**9)
            y = cagerGrid[r] * np.sin(np.arccos(cagexGL[TH])) * np.sin(cagephiGrid[PHI]) * (10**9)
            z = cagerGrid[r] * cagexGL[TH] * (10**9)

            positTempx = x + rotOx
            positTempy = y + rotOy
            positTempz = z + rotOz
            temprsq =(positTempx[:,:,:,np.newaxis] - xdat[np.newaxis,np.newaxis,np.newaxis,:])**2 + (positTempy[:,:,:,np.newaxis] - ydat[np.newaxis,np.newaxis,np.newaxis,:])**2 + (positTempz[:,:,:,np.newaxis] - zdat[np.newaxis,np.newaxis,np.newaxis,:])**2
            tempr6 = temprsq*temprsq*temprsq
            tempr12 = tempr6*tempr6
            v += np.sum((sigO12ep/tempr12) - (sigO6ep/tempr6),axis=-1)
            
            positTempx = x + roth1x
            positTempy = y + roth1y
            positTempz = z + roth1z
            temprsq =(positTempx[:,:,:,np.newaxis] - xdat[np.newaxis,np.newaxis,np.newaxis,:])**2 + (positTempy[:,:,:,np.newaxis] - ydat[np.newaxis,np.newaxis,np.newaxis,:])**2 + (positTempz[:,:,:,np.newaxis] - zdat[np.newaxis,np.newaxis,np.newaxis,:])**2
            tempr6 = temprsq*temprsq*temprsq
            tempr12 = tempr6*tempr6
            v += np.sum((sigH12ep/tempr12) - (sigH6ep/tempr6),axis=-1)

            positTempx = x + roth2x
            positTempy = y + roth2y
            positTempz = z + roth2z
            temprsq =(positTempx[:,:,:,np.newaxis] - xdat[np.newaxis,np.newaxis,np.newaxis,:])**2 + (positTempy[:,:,:,np.newaxis] - ydat[np.newaxis,np.newaxis,np.newaxis,:])**2 + (positTempz[:,:,:,np.newaxis] - zdat[np.newaxis,np.newaxis,np.newaxis,:])**2
            tempr6 = temprsq*temprsq*temprsq
            tempr12 = tempr6*tempr6
            v += np.sum((sigH12ep/tempr12) - (sigH6ep/tempr6),axis=-1)
            temp[TH][PHI] = v
    return temp
##########################################################################################





#M A I N

if __name__ == '__main__':    
    print(__name__)
    ###########################
    ## USER INPUT VARIABLES  ##
    ##################################
    Jmax=int(sys.argv[1])            #  
    Nmax = int(sys.argv[2])          # 
    angleNum = int(sys.argv[3])      #
    cageGridNum = int(sys.argv[4])       #
    cageAngleNum = int(sys.argv[5])  #
    cageMaxR = 0.08#35#2#4#235#float(sys.argv[6]) #in nm
    ##################################
    cage = 'A'                       #
    ##################################
    print("Jmax: ", Jmax)
    print("Nmax: ", Nmax)
    print("angleNum: ", angleNum)
    print("cageMaxR: ", cageMaxR)
    print("r pts: ", cageGridNum)
    print("cageAngleNum: ", cageAngleNum)
    
    #############
    # CONSTANTS #
    #############
    
    #########################
    ## UNIVERSAL CONSTANTS ##
    ######################################
    h = 6.62607015*(10**-34)
    hbar = h/(2.*np.pi)
    clight = 299792458.
    jpcm = h*clight*100.
    #jpcm = 1.9863*(10**-23)
    ######################################
    
    #JOULES TO 1/cm  ==>  E = hc/lam   E / hc 
    
    #########################
    ## MOLECULAR CONSTANTS ##
    ##############################################################
    #NA = 6.022140857*(10**23)
    NA = 6.02214076*(10**23)
    massh2o = (18.01056/(NA*1000.))#
    
    mO = 15.9994
    mH = 1.008
    mH2O = 2.*mH + mO
                                                                 #       ####################
    Ah2o= 27.877 #cm-1                                           #        ##
    Bh2o= 14.512 #cm-1                                           #         ##
    Ch2o= 9.285 #cm-1                                            #
                                                                 #
    Ost = np.array([0., 0., -0.006563807])                       #
    H1st = np.array([0.07575, 0., 0.052086193])                  #
    H2st = np.array([-0.07575, 0., 0.052086193])                 #
    
    H2OCoM = (mH*(H1st+H2st) + mO*Ost)/mH2O
    
    print("H2O CoM: ")
    print(H2OCoM)
    
    OMF = Ost - H2OCoM
    H1MF = H1st - H2OCoM
    H2MF = H2st - H2OCoM
    ##############################################################
    
    
    ###########################
    ## INTERACTION CONSTANTS ##
    #############################################################       change these parameters based off of potential model
    omega = 3.235147426*(10**13) #hz                            #       -omega is fit from the classic potential curve
    # 4.184*83.59539 = 349.76311176
    # 1 kcal = 4.184 kJ
    # from my calc:  349.75509                            
    # from all those vals: 349.7550882
    epsilonOC = 0.1039*349.7550882#4.184*83.59539 #cm-1    
    sigmaOC = 0.3372 #nm                   
    epsilonHC = 0.0256*349.7550882#4.184*83.59539 #cm-1      
    sigmaHC = 0.264 #nm                     
                           
    sigmaOCsix = sigmaOC**6.
    sigmaOCtwelve = sigmaOC**12.
    sigmaHCsix = sigmaHC**6.
    sigmaHCtwelve = sigmaHC**12.
                                                            
    sigO6ep = 4.*epsilonOC*sigmaOCsix                        
    sigO12ep = 4.*epsilonOC*sigmaOCtwelve                     
    sigH6ep = 4.*epsilonHC*sigmaHCsix                          
    sigH12ep = 4.*epsilonHC*sigmaHCtwelve                       
    
    thetaNum = int(angleNum+1)                                           
    xGL,wGL = np.polynomial.legendre.leggauss(thetaNum)              
    phixiGridPts = np.linspace(0,2*np.pi,angleNum, endpoint=False)  
    dphixi = 2.*np.pi / angleNum                                 
    
    cagexGL, cagewGL = np.polynomial.legendre.leggauss(cageAngleNum)
    cagerGrid = np.linspace(0.,cageMaxR, cageGridNum)
    cagerGrid*=(10**-9)
    print(cagerGrid)
    dr = (cagerGrid[1] - cagerGrid[0])#*(10**-9)
    print("dr:", dr)
    
    cagephiGrid = np.linspace(0.,2*np.pi,cageAngleNum,endpoint=False)
    
    dphi = 2.*np.pi / cageAngleNum
    
    
    
    #########################################################################
    # Calculate the rotation matrix at every angular grid point             #
    # in order to rotate the MFF into the SFF, shape; [phi, xi, theta, 3,3] #
    #########################################################################
    rotmat = np.zeros((len(xGL),len(phixiGridPts),len(phixiGridPts),3,3),float)
    
    for xgls in range(len(xGL)):
        for phi in range(len(phixiGridPts)):
            for xi in range(len(phixiGridPts)):
                rotmat[xgls,phi,xi,:,:] = np.array( [ [ np.cos(phixiGridPts[phi]) * np.cos(np.arccos(xGL[xgls])) * np.cos( phixiGridPts[xi]) - np.sin(phixiGridPts[phi]) * np.sin(phixiGridPts[xi]), -np.cos( phixiGridPts[phi] ) * np.cos(np.arccos( xGL[xgls] ) ) * np.sin( phixiGridPts[xi] ) - np.sin( phixiGridPts[phi] ) * np.cos( phixiGridPts[xi] ) , np.cos( phixiGridPts[phi] ) * np.sin( np.arccos( xGL[xgls] ) )], [np.sin( phixiGridPts[phi] ) * np.cos( np.arccos( xGL[xgls] ) ) * np.cos(phixiGridPts[xi] ) + np.cos( phixiGridPts[phi] ) * np.sin(phixiGridPts[xi] ), -np.sin( phixiGridPts[phi] ) * np.cos( np.arccos( xGL[xgls] ) ) * np.sin( phixiGridPts[xi] ) + np.cos( phixiGridPts[phi] ) * np.cos( phixiGridPts[xi] ), np.sin(phixiGridPts[phi]) * np.sin(np.arccos(xGL[xgls]))], [-np.sin(np.arccos(xGL[xgls]))*np.cos(phixiGridPts[xi]), np.sin(np.arccos(xGL[xgls]))*np.sin(phixiGridPts[xi]), xGL[xgls] ]])  
    
    print(rotmat.shape)
    # rotate the MFF components for the new coordinates          
    #####################################################################################
    #rotO = np.inner(rotmat.swapaxes(3,4), OMF)
    #rotH1 = np.inner(rotmat.swapaxes(3,4),H1MF)
    #rotH2 = np.inner(rotmat.swapaxes(3,4),H2MF)
    
    rotO = np.inner(rotmat, OMF)
    rotH1 = np.inner(rotmat,H1MF)
    rotH2 = np.inner(rotmat,H2MF)
    # Extract the cartesian components#
    # of the rotated MFF atoms        # 
    rotOx = rotO[:,:,:,0]             #
    rotOy = rotO[:,:,:,1]#      O     #
    rotOz = rotO[:,:,:,2]             #
                                      #
    roth1x = rotH1[:,:,:,0]           #
    roth1y = rotH1[:,:,:,1]#   H1     #
    roth1z = rotH1[:,:,:,2]           #
                                      #
    roth2x = rotH2[:,:,:,0]           #
    roth2y = rotH2[:,:,:,1]#   H2     #
    roth2z = rotH2[:,:,:,2]           #
    ###################################
    
    
    
    JKM = int(((2*Jmax +1) * (2*Jmax + 2) * (2*Jmax + 3))/6) 
    
    if Jmax%2 ==0:
        JKeM = int((JKM+Jmax+1)/2)
        JKoM = int(JKM-JKeM)
    else:
        JKoM = int((JKM+Jmax+1)/2)
        JKeM = int(JKM-JKoM)
    
    Nnlm = int(((Nmax + 1) * (Nmax + 2) * (Nmax + 3))/6) 
    
    
    shortJKeM = 0
    shortJKoM = 0
    for J in range(Jmax+1):
        for K in range(0,J+1,2):
            shortJKeM+= 2*J+1
        for K in range(1,J+1,2):
            shortJKoM+=2*J+1
    ####################################################################
    #
    #   COUNT THE NUMBER OF BASIS STATES IN EACH BLOCK (Au, Ag, Bu, Bg)
    #
    
    #A (u+g) (K-even, eps=0 or 1)
    Agcounter = 0
    Aucounter = 0
    for n in range(Nmax+1):
        lmin = n%2
        for l in range(lmin,n+1,2):
            for m in range(-l,l+1):
                for J in range(Jmax+1):
                    Gparity = J + l
                    Uparity = J + l + 1
                    if (Gparity%2)==0:
                        for K in range(0,J+1,2):
                            for M in range(-J,J+1):
                                Agcounter+=1
                        
                    else:
                        for K in range(2,J+1,2):
                            for M in range(-J,J+1):
                                Agcounter+=1
    
                    if (Uparity%2)==0:
                        for K in range(0,J+1,2):
                            for M in range(-J,J+1):
                                Aucounter+=1
                    else:
                        for K in range(2,J+1,2):
                            for M in range(-J,J+1):
                                Aucounter+=1
    
    
    #B (u+g) (K-odd, eps=0 or 1)
    Bucounter = 0
    Bgcounter = 0
    for n in range(Nmax+1):
        lmin = n%2
        for l in range(lmin,n+1,2):
            for m in range(-l,l+1):
                for J in range(Jmax+1):
                    for K in range(1,J+1,2):
                        for M in range(-J,J+1):
                            Bucounter+=1
                            Bgcounter+=1
    
    
    AuQuantumNumList = np.zeros((Aucounter,7),int)
    AgQuantumNumList = np.zeros((Agcounter,7),int)
    BuQuantumNumList = np.zeros((Bucounter,7),int)
    BgQuantumNumList = np.zeros((Bgcounter,7),int)
    NnlmQuantumNumList = np.zeros((Nnlm,3),int)
    JKeMQuantumNumList = np.zeros((JKeM,3),int)
    JKoMQuantumNumList = np.zeros((JKoM,3),int)
    
    shortJKeMQuantumNumList = np.zeros((shortJKeM,3),int)
    shortJKoMQuantumNumList = np.zeros((shortJKoM,3),int)
    
    Aureverse={}
    Agreverse={}
    Bureverse={}
    Bgreverse={}
    Nnlmreverse={}
    JKeMreverse={}
    JKoMreverse={}
    
    shortJKeMreverse={}
    shortJKoMreverse={}
    
    ####################################################################
    #
    #  Determine the quantum numbers |Nxyz>|JKM,eps>  for each block 
    #
    
    
    #A (u+g) (K-even, eps=0 or 1)
    Agcounter=0
    Aucounter=0
    for n in range(Nmax+1):
        lmin = n%2
        for l in range(lmin,n+1,2):
            for m in range(-l,l+1):
                for J in range(Jmax+1):
                    Gparity = J + l
                    Uparity = J + l + 1
                    if (Gparity%2)==0:
                        for K in range(0,J+1,2):
                            PoM = (Gparity+K)%2
                            for M in range(-J,J+1):
                                AgQuantumNumList[Agcounter,0]=n
                                AgQuantumNumList[Agcounter,1]=l
                                AgQuantumNumList[Agcounter,2]=m
                                AgQuantumNumList[Agcounter,3]=J
                                AgQuantumNumList[Agcounter,4]=K
                                AgQuantumNumList[Agcounter,5]=M
                                AgQuantumNumList[Agcounter,6]=PoM
                                Agreverse[(n,l,m,J,K,M)]=Agcounter
                                Agcounter+=1
    
                        
                    else:
                        for K in range(2,J+1,2):
                            PoM = (Gparity+K)%2
                            for M in range(-J,J+1):
                                AgQuantumNumList[Agcounter,0]=n
                                AgQuantumNumList[Agcounter,1]=l
                                AgQuantumNumList[Agcounter,2]=m
                                AgQuantumNumList[Agcounter,3]=J
                                AgQuantumNumList[Agcounter,4]=K
                                AgQuantumNumList[Agcounter,5]=M
                                AgQuantumNumList[Agcounter,6]=PoM
                                Agreverse[(n,l,m,J,K,M)]=Agcounter
                                Agcounter+=1
    
                    if (Uparity%2)==0:
                        for K in range(0,J+1,2):
                            PoM = (Uparity+K)%2
                            for M in range(-J,J+1):
                                AuQuantumNumList[Aucounter,0]=n
                                AuQuantumNumList[Aucounter,1]=l
                                AuQuantumNumList[Aucounter,2]=m
                                AuQuantumNumList[Aucounter,3]=J
                                AuQuantumNumList[Aucounter,4]=K
                                AuQuantumNumList[Aucounter,5]=M
                                AuQuantumNumList[Aucounter,6]=PoM
                                Aureverse[(n,l,m,J,K,M)]=Aucounter
                                Aucounter+=1
                    else:
                        for K in range(2,J+1,2):
                            PoM = (Uparity+K)%2
                            for M in range(-J,J+1):
                                AuQuantumNumList[Aucounter,0]=n
                                AuQuantumNumList[Aucounter,1]=l
                                AuQuantumNumList[Aucounter,2]=m
                                AuQuantumNumList[Aucounter,3]=J
                                AuQuantumNumList[Aucounter,4]=K
                                AuQuantumNumList[Aucounter,5]=M
                                AuQuantumNumList[Aucounter,6]=PoM
                                Aureverse[(n,l,m,J,K,M)]=Aucounter
                                Aucounter+=1
    
    
    
    #B (g+u) (K-odd, eps=0 or 1)
    
    Bgcounter = 0
    Bucounter = 0
    for n in range(Nmax+1):
        lmin = n%2
        for l in range(lmin,n+1,2):
            for m in range(-l,l+1):
                for J in range(Jmax+1):
                    for K in range(1,J+1,2):
                        PoMG = (J + K + l)%2
                        PoMU = (J + K + l + 1)%2
                        for M in range(-J,J+1):
                            BuQuantumNumList[Bucounter,0]=n
                            BuQuantumNumList[Bucounter,1]=l
                            BuQuantumNumList[Bucounter,2]=m
                            BuQuantumNumList[Bucounter,3]=J
                            BuQuantumNumList[Bucounter,4]=K
                            BuQuantumNumList[Bucounter,5]=M
                            BuQuantumNumList[Bucounter,6]=PoMU
                            Bureverse[(n,l,m,J,K,M)]=Bucounter
                            Bucounter+=1
    
                            BgQuantumNumList[Bgcounter,0]=n
                            BgQuantumNumList[Bgcounter,1]=l
                            BgQuantumNumList[Bgcounter,2]=m
                            BgQuantumNumList[Bgcounter,3]=J
                            BgQuantumNumList[Bgcounter,4]=K
                            BgQuantumNumList[Bgcounter,5]=M
                            BgQuantumNumList[Bgcounter,6]=PoMG
                            Bgreverse[(n,l,m,J,K,M)]=Bgcounter
                            Bgcounter+=1
    
    print("Nnlm JKeM:  ", Nnlm*JKeM)
    print("Au + Ag:  ", Aucounter+Agcounter)
    print("Nnlm JKoM:  ", Nnlm*JKoM)
    print("Bu + Bg:  ", Bucounter+Bgcounter)
    print("Au :", Aucounter)
    print("Ag :", Agcounter)
    print("Bu :", Bucounter)
    print("Bg :", Bgcounter)
    sys.stdout.flush()
    #determine the quantum numbers in the polyad truncated 3x coupled H.O.  basis
    nlmcounter = 0
    for n in range(Nmax+1):
        lmin = n%2
        for l in range(lmin,n+1,2):
            for m in range(-l,l+1):
                NnlmQuantumNumList[nlmcounter,0] = n
                NnlmQuantumNumList[nlmcounter,1] = l
                NnlmQuantumNumList[nlmcounter,2] = m
                Nnlmreverse[(n,l,m)]=nlmcounter
                nlmcounter+=1
    print(Nnlm, nlmcounter)
    print("!#!@#!@#!@#")
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
    print(jtempcounter)
    #odd
    jtempcounter = 0
    for J in range(Jmax+1):
        if J%2==0:
            for K in range(-J+1,J,2):
                print(K)
                for M in range(-J,J+1):
                    JKoMQuantumNumList[jtempcounter,0]=J
                    JKoMQuantumNumList[jtempcounter,1]=K
                    JKoMQuantumNumList[jtempcounter,2]=M
                    JKoMreverse[(J,K,M)]=jtempcounter
                    jtempcounter+=1
        else:
            for K in range(-J,J+1,2):
                print(K)
                for M in range(-J,J+1):
                    JKoMQuantumNumList[jtempcounter,0]=J
                    JKoMQuantumNumList[jtempcounter,1]=K
                    JKoMQuantumNumList[jtempcounter,2]=M
                    JKoMreverse[(J,K,M)]=jtempcounter
                    jtempcounter+=1
    
    jtempcounter=0
    for J in range(Jmax+1):
        for K in range(0,J+1,2):
            for M in range(-J,J+1):
                shortJKeMQuantumNumList[jtempcounter,0]=J
                shortJKeMQuantumNumList[jtempcounter,1]=K
                shortJKeMQuantumNumList[jtempcounter,2]=M
                shortJKeMreverse[(J,K,M)]=jtempcounter
                jtempcounter+=1
    
    jtempcounter=0
    for J in range(Jmax+1):
        for K in range(1,J+1,2):
            for M in range(-J,J+1):
                shortJKoMQuantumNumList[jtempcounter,0]=J
                shortJKoMQuantumNumList[jtempcounter,1]=K
                shortJKoMQuantumNumList[jtempcounter,2]=M
                shortJKoMreverse[(J,K,M)]=jtempcounter
                jtempcounter+=1
    #keven = A, kodd = B, eps = 0  g, eps 1 = u
    #tempKeU = 0
    #tempKeG = 0
    
   # for J in range(Jmax+1):
   #     for k in range(0,(J+1),2):
   #         if k==0:
   ##             tempKeG += 2*J+1
   #         else:
   #             tempKeU += 2*J+1
   #             tempKeG += 2*J+1
   # 
   # KevenU = tempKeU
   # KevenG = tempKeG
   # KoddU = JKoM / 2
   # KoddG = JKoM / 2
    
    HAg = np.zeros((Agcounter,Agcounter),complex)
    HAu = np.zeros((Aucounter,Aucounter),complex)
    HBg = np.zeros((Bgcounter,Bgcounter),complex)
    HBu = np.zeros((Bucounter,Bucounter),complex)

    
    dJKeM = np.zeros((shortJKeM, len(xGL)), complex)
    invdJKeM = np.zeros((shortJKeM, len(xGL)), complex)

    KJKeM = np.zeros( (shortJKeM, angleNum),complex)
    invKJKeM = np.zeros((shortJKeM, angleNum),complex)

    MJKeM = np.zeros( (shortJKeM, angleNum),complex)


    
    dJKoM = np.zeros((shortJKoM, len(xGL)), complex)
    invdJKoM = np.zeros((shortJKoM, len(xGL)), complex)

    KJKoM = np.zeros((shortJKoM, angleNum),complex)
    invKJKoM = np.zeros((shortJKoM, angleNum),complex)

    MJKoM = np.zeros((shortJKoM, angleNum),complex)


    
    sys.stdout.flush()
    #for s in range(JKeM):
    for s in range(shortJKeM):
        if shortJKeMQuantumNumList[s,1] == 0:
            Nk = 0.5
        else:
            Nk = 1./np.sqrt(2.)
        for th in range(len(xGL)):
    
            dJKeM[s,th] = np.sqrt( (2.*shortJKeMQuantumNumList[s,0]+1) / (8.*np.pi**2) ) * littleD(shortJKeMQuantumNumList[s,0],shortJKeMQuantumNumList[s,2],shortJKeMQuantumNumList[s,1],np.arccos(xGL[th])) * np.sqrt(wGL[th])

            #cdJKeM[s,th] = (-1.0)**(shortJKeMQuantumNumList[s,1] - shortJKeMQuantumNumList[s,2]) * np.sqrt( (2.*shortJKeMQuantumNumList[s,0]+1) / (8.*np.pi**2) ) * littleD(shortJKeMQuantumNumList[s,0],shortJKeMQuantumNumList[s,2],shortJKeMQuantumNumList[s,1],np.arccos(xGL[th])) * np.sqrt(wGL[th])
    
            invdJKeM[s,th] = ((-1.0)**(shortJKeMQuantumNumList[s,0]))*np.sqrt( (2.*shortJKeMQuantumNumList[s,0]+1) / (8.*np.pi**2) ) * littleD(shortJKeMQuantumNumList[s,0],shortJKeMQuantumNumList[s,2],-shortJKeMQuantumNumList[s,1],np.arccos(xGL[th])) * np.sqrt(wGL[th])

            #invcdJKeM[s,th] = ((-1.0)**(shortJKeMQuantumNumList[s,0] +shortJKeMQuantumNumList[s,1] - shortJKeMQuantumNumList[s,2]))*np.sqrt( (2.*shortJKeMQuantumNumList[s,0]+1) / (8.*np.pi**2) ) * littleD(shortJKeMQuantumNumList[s,0],shortJKeMQuantumNumList[s,2],-shortJKeMQuantumNumList[s,1],np.arccos(xGL[th])) * np.sqrt(wGL[th])
    
        for ph in range(angleNum):
            KJKeM[s,ph] = np.exp(-1j*phixiGridPts[ph]*shortJKeMQuantumNumList[s,1]) * np.sqrt(dphixi) * Nk
            invKJKeM[s,ph] = ((-1.0)**(shortJKeMQuantumNumList[s,1])) * np.exp(1j*phixiGridPts[ph]*shortJKeMQuantumNumList[s,1]) * np.sqrt(dphixi) * Nk

            MJKeM[s,ph] = np.exp(-1j*phixiGridPts[ph]*shortJKeMQuantumNumList[s,2]) * np.sqrt(dphixi)
    
    for s in range(shortJKoM):
        Nk = 1./np.sqrt(2.)
        for th in range(len(xGL)):
            dJKoM[s,th] = np.sqrt( (2.*shortJKoMQuantumNumList[s,0]+1) / (8.*np.pi**2) ) * littleD(shortJKoMQuantumNumList[s,0],shortJKoMQuantumNumList[s,2],shortJKoMQuantumNumList[s,1],np.arccos(xGL[th])) * np.sqrt(wGL[th])

            #cdJKoM[s,th] = (-1.0)**(shortJKoMQuantumNumList[s,1] - shortJKoMQuantumNumList[s,2])*np.sqrt( (2.*shortJKoMQuantumNumList[s,0]+1) / (8.*np.pi**2) ) * littleD(shortJKoMQuantumNumList[s,0],shortJKoMQuantumNumList[s,2],shortJKoMQuantumNumList[s,1],np.arccos(xGL[th])) * np.sqrt(wGL[th])

            invdJKoM[s,th] = ((-1.0)**(shortJKoMQuantumNumList[s,0]))*np.sqrt( (2.*shortJKoMQuantumNumList[s,0]+1) / (8.*np.pi**2) ) * littleD(shortJKoMQuantumNumList[s,0],shortJKoMQuantumNumList[s,2],-shortJKoMQuantumNumList[s,1],np.arccos(xGL[th])) * np.sqrt(wGL[th])

            #invcdJKoM[s,th] = ((-1.0)**(shortJKoMQuantumNumList[s,0] + shortJKoMQuantumNumList[s,1] + shortJKoMQuantumNumList[s,2] ) )*np.sqrt( (2.*shortJKoMQuantumNumList[s,0]+1) / (8.*np.pi**2) ) * littleD(shortJKoMQuantumNumList[s,0],shortJKoMQuantumNumList[s,2],-shortJKoMQuantumNumList[s,1],np.arccos(xGL[th])) * np.sqrt(wGL[th])
        for ph in range(angleNum):
            KJKoM[s,ph] = np.exp(-1j*phixiGridPts[ph]*shortJKoMQuantumNumList[s,1]) * np.sqrt(dphixi) * Nk
            invKJKoM[s,ph] = ((-1.0)**(shortJKoMQuantumNumList[s,1]))*np.exp(1j*phixiGridPts[ph]*shortJKoMQuantumNumList[s,1]) * np.sqrt(dphixi) * Nk

            MJKoM[s,ph] = np.exp(-1j*phixiGridPts[ph]*shortJKoMQuantumNumList[s,2]) * np.sqrt(dphixi)
    
    #############################################################################################################
    
    
    ##################################################################
    # PRECALCULATE 3D ISO H.O. PSI_nlm's to use in V matrix element  #
    ##################################################################
    nu = (massh2o*omega)/(2.*hbar)
    
    doubleFactorial = np.zeros((4*Nmax+1),float)
    for n in range(1,4*Nmax+1):
        if n%2 ==0:
            temp=1.
            for k in range(1,int(n/2+1)):
                temp*=2*k
            doubleFactorial[n]=temp
        if n%2==1:
            temp=1.
            for k in range(1,int((n+1)/2+1)):
                temp*=(2*k-1)
            doubleFactorial[n]=temp
    doubleFactorial[0] = 1.0
    print("double factorial: n=0", doubleFactorial[0])
    print("double factorial: n=1", doubleFactorial[1])
    print("double factorial: n=2", doubleFactorial[2])
    print("double factorial: n=3", doubleFactorial[3])
    print("double factorial: n=4", doubleFactorial[4])
    
    #Normalization constant Nkl
    print("nkl")
    Nkl =np.zeros((int(Nmax/2)+1, Nmax+1))
    for k in range(int(Nmax/2)+1):
        for l in range(Nmax+1):
            print(k, l)
            Nkl[k,l] = np.sqrt(  np.sqrt( (2.*(nu**3))/np.pi) * (2**(k+2.*l+3) * np.math.factorial(k) * nu**l)/ doubleFactorial[2*k + 2*l+1])
    
    #Generalized Laguerre Polynomials Lkl(k,l+1/2)
    Lkl = np.zeros((int(Nmax/2)+1,Nmax+1,cageGridNum),float)
    
    for l in range(Nmax+1):
        for r in range(cageGridNum):
            Lkl[0,l,r] = 1.
    
    for l in range(Nmax+1):
        for r in range(cageGridNum):
            rArg = 2.*nu*(cagerGrid[r]**2)
            Lkl[1,l,r] = 1. + l +0.5 -rArg
    

    for l in range(Nmax+1):
        for k in range(1,int(Nmax/2)):
            for r in range(cageGridNum):
                rArg = 2.*nu*(cagerGrid[r]**2)
                Lkl[k+1,l,r] = ((2*k + 1.5 + l - rArg)*Lkl[k,l,r] - (k+l+0.5)*Lkl[k-1,l,r] )/(k+1)
    
    temp = 0
    for l in range(Nmax+1):
        temp+= 2*l+1
    
    PYlmNum = temp
    #calculate the associated legendre polynomials Plm and sphereical harmonics Ylm
    PYlmQuantumNumList = np.zeros((PYlmNum,2),int)
    PYlmReverse = {}
    
    counter=0
    for l in range(Nmax+1):
        for m in range(-l,l+1):
            PYlmQuantumNumList[counter,0]=l
            PYlmQuantumNumList[counter,0]=m
            PYlmReverse[(l,m)]=counter
            print( "l:",l,"  m:",m, "   counter:",counter)
            counter+=1
    
    Plm = np.zeros((PYlmNum,len(cagexGL)))
    Plm[PYlmReverse[0,0],:] = 1
    
    for l in range(Nmax):
        for th in range(len(cagexGL)):
            Plm[PYlmReverse[l+1,l+1],th] = -(2.* l+1) * Plm[PYlmReverse[l,l],th] * np.sqrt(1.- cagexGL[th]**2)
    
    for l in range(Nmax):
        for th in range(len(cagexGL)):
            Plm[PYlmReverse[l+1,l],th] = cagexGL[th] * (2.*l+1)*Plm[PYlmReverse[l,l],th]
    
    for m in range(Nmax-1):
        for l in range(m+1,Nmax):
            for th in range(len(cagexGL)):
                Plm[PYlmReverse[l+1,m],th] = ((2.*l+1) * cagexGL[th] *Plm[PYlmReverse[l,m],th] - (l+m)*Plm[PYlmReverse[l-1,m],th])/(l-m+1.)
    
    #for l in range(1,Nmax+1):
    #    for m in range(-l,0):
     #       for th in range(len(cagexGL)):
      #          Plm[PYlmReverse[l,m],th] = ((-1.0)**(m))*(np.math.factorial(l-m) / np.math.factorial(l+m)) * Plm[PYlmReverse[l,-m],th]
   






 
    Ylm = np.zeros((PYlmNum,len(cagexGL),len(cagephiGrid)),complex)
    Ylmconj = np.zeros((PYlmNum,len(cagexGL),len(cagephiGrid)),complex)
    
    for l in range(Nmax+1):
        for m in range(l+1):
            for th in range(len(cagexGL)):
                for ph in range(len(cagephiGrid)):
                    Ylm[PYlmReverse[l,m], th,ph] = ((-1.)**(m)) *np.sqrt( ((2.*l+1 )*(np.math.factorial(l-m))) /  (2.*np.math.factorial(l+m)))  * Plm[PYlmReverse[l,m],th] * np.sqrt(1./(2.*np.pi)) * np.exp(1j*m*cagephiGrid[ph])
                    #Ylm[PYlmReverse[l,m], th,ph] = ((-1.)**(m)) * np.sqrt( ((2.*l+1 )*(np.math.factorial(l-m))) /  (2.*np.math.factorial(l+m)))  * Plm[PYlmReverse[l,m],th] * np.sqrt(1./(2.*np.pi)) * np.exp(1j*m*cagephiGrid[ph])
    
    
    for l in range(Nmax+1):
        for m in range(-l,0):
            for th in range(len(cagexGL)):
                for ph in range(len(cagephiGrid)):
                    Ylm[PYlmReverse[l,m], th,ph] = np.sqrt( ((2.*l+1 )*(np.math.factorial(l-np.abs(m)))) /  (2.*np.math.factorial(l+np.abs(m))))  * Plm[PYlmReverse[l,np.abs(m)],th] * np.sqrt(1./(2.*np.pi)) * np.exp(1j*m*cagephiGrid[ph])
    
    ########## MAYBE -m HERE?
    
    for l in range(Nmax+1):
        for m in range(-l,l+1):
            for th in range(len(cagexGL)):
                for ph in range(len(cagephiGrid)):
                    Ylmconj[PYlmReverse[l,m],th,ph] = ((-1.)**(m)) * Ylm[PYlmReverse[l,-m],th,ph] 
    #####################################################################
    #precalculate the |Nx>|Ny>|Nz> basis function in (x,y,z) grid       #
    #for the <Nx|<Ny|<Nz||Vjkmj'k'm'(x,y,z)|Nx'>|Ny'>|Nz'> integration  #
    #store as <Nxyz|  since strange shape due to polyad truncation      #
    #####################################################################
    PSInlm = np.zeros((Nnlm, cageGridNum, cageAngleNum, cageAngleNum),complex)                                     
    PSInlminv = np.zeros((Nnlm, cageGridNum*cageAngleNum* cageAngleNum),complex)                                     
    PSIconjnlm = np.zeros((Nnlm, cageGridNum, cageAngleNum, cageAngleNum),complex)                                     
    PSIconjnlminv = np.zeros((Nnlm, cageGridNum* cageAngleNum* cageAngleNum),complex)                                     
    #print "Nxyz: ", Nxyz      
    npos = 0                                                                                        
    for n in range(Nnlm):
        k = int((NnlmQuantumNumList[n,0] - NnlmQuantumNumList[n,1]) / 2)
        print(k)
        for r in range(cageGridNum):                                                            
            for TH in range(cageAngleNum):                                                        
                for PH in range(cageAngleNum):                                                    
                    PSInlm[n,r,TH,PH] = np.sqrt(dphi*cagewGL[TH]*dr) * cagerGrid[r]**(NnlmQuantumNumList[n,1]+1) *  Nkl[k, NnlmQuantumNumList[n,1]] * np.exp(-nu * cagerGrid[r]**2) * Lkl[k,NnlmQuantumNumList[n,1],r] * Ylm[PYlmReverse[NnlmQuantumNumList[n,1],NnlmQuantumNumList[n,2]], TH, PH]

                    PSIconjnlm[n,r,TH,PH] = np.sqrt(dphi*cagewGL[TH]*dr)*cagerGrid[r]**(NnlmQuantumNumList[n,1]+1) * Nkl[k, NnlmQuantumNumList[n,1]] * np.exp(-nu * cagerGrid[r]**2) * Lkl[k,NnlmQuantumNumList[n,1],r] * Ylmconj[PYlmReverse[NnlmQuantumNumList[n,1],NnlmQuantumNumList[n,2]], TH, PH]
    
    PSInlm = np.reshape(PSInlm, (Nnlm, cageGridNum*cageAngleNum*cageAngleNum), order='C')                       
    PSIconjnlm = np.reshape(PSIconjnlm, (Nnlm, cageGridNum*cageAngleNum*cageAngleNum), order='C')           
    
    for n in range(Nnlm):
        PSInlminv[n,:] = ((-1.0)**NnlmQuantumNumList[n,1]) * PSInlm[n,:]
        PSIconjnlminv[n,:] = ((-1.0)**NnlmQuantumNumList[n,1]) * PSIconjnlm[n,:]
    #test = np.sum(PSInlm[:,np.newaxis,:]*PSIconjnlm[np.newaxis,:,:],axis=2)
    #print("nlm test")
    #print(test.shape)
    #print(np.sum(test), np.sum(test-np.diag(np.diag(test)))) 
    #test2 = np.sum(PSInlminv[:,np.newaxis,:] * PSIconjnlminv[np.newaxis,:,:],axis=2)
    #print("inv nlm test")
    #print(test2.shape)
    #print(np.sum(test2), np.sum(test2-np.diag(np.diag(test2))))
    #exit()#  june 23 : this is orthogonal 
    #nxyzbasis = nxyzvec[:,np.newaxis,:] * nxyzvec[np.newaxis,:,:]                                   
    ###################################################################################
    
    basisSetupTime = datetime.now()
    
    #####################################################
    # Load co-ordinates for external potential LJ sites #
    #####################################################   
    fop = open('geometries/'+str(cage),'r')                      #    - my coordinate file is in angstrom but
    atomNum = int(fop.readline())                       #     calculation units in nm so divide by 10
    xdat = np.zeros((atomNum),float)                    #    - first line has number of atoms to read in
    ydat = np.zeros((atomNum),float)                    #    - first(0th) column is thrown out, second[1], third[2] and fourth[3] contain x,y,z data
    zdat = np.zeros((atomNum),float)                    # 
    for site in range(atomNum):                         #
        data = fop.readline().split()                   #
        xdat[site] = float(data[1])/10.0 #nm            #
        ydat[site] = float(data[2])/10.0 #nm            #
        zdat[site] = float(data[3])/10.0 #nm            #
    fop.close()                                         #
    #####################################################
    
    
    
    vStart = datetime.now()
    ####################################################
    # distribute potential calculation over processors #
    ##########################################################################################

    v6d = np.zeros((cageGridNum,cageAngleNum,cageAngleNum, thetaNum,angleNum,angleNum))
    for z in range(cageGridNum):
        print("r: ,",z)
        v6d[z]=calcvr(z)
    #v6d *= 0.0 
    sys.stdout.flush()
   # p.close()                                                                                #
   # p.join()                                                                                 #
                                                                                             #
    #v6d = np.array(v6d)                                                                      #
    print(v6d.shape, "v6d shape")                                                             #
    print(np.sum(v6d))
    v6d = np.reshape(v6d, (cageGridNum*cageAngleNum*cageAngleNum, thetaNum,angleNum,angleNum), order='C')  #
    print("V-6D done")                                                                        #
                                                                                             #
    ########################################################################################## 
    vEnd = datetime.now()
    
    
    ########################################################################################################################################
    
    
    ############################################
    #function: calcvel -- calculate v- element #
    #computes the quadrature  vjkmj'k'm'(x,y,z)# 
    #from <jkm(w)|V(xyz,w)|j'k'm'(w)>          #
    ############################################
    vAg = np.zeros((Agcounter,Agcounter),complex)
    vAu = np.zeros((Aucounter,Aucounter),complex)
    vBg = np.zeros((Bgcounter,Bgcounter),complex)
    vBu = np.zeros((Bucounter,Bucounter),complex)
    
    ########################################
    #
    # n - __QuantumNumList[i,0]
    # l - __QuantumNumList[i,1]
    # m - __QuantumNumList[i,2]
    # J - __QuantumNumList[i,3]
    # K - __QuantumNumList[i,4]
    # M - __QuantumNumList[i,5]
    #
    ########################################
    # Calculate the V element for Au block
    ########################################
    shortJKeMrange = range(shortJKeM)                         
    shortJKoMrange = range(shortJKoM)
    
    print(shortJKeM)
    print(shortJKoM)
    
    

    #eEEbasisuse = KJKeM[:,np.newaxis,np.newaxis,np.newaxis,:]*np.conj(KJKeM[np.newaxis,:,np.newaxis,np.newaxis,:]) * MJKeM[:,np.newaxis,np.newaxis,:,np.newaxis] * np.conj(MJKeM[np.newaxis,:,np.newaxis,:,np.newaxis]) * dJKeM[:,np.newaxis,:,np.newaxis,np.newaxis] * dJKeM[np.newaxis,:,:,np.newaxis,np.newaxis]
    eEEbasisuse = KJKeM[:,np.newaxis,np.newaxis,np.newaxis,:] * MJKeM[:,np.newaxis,np.newaxis,:,np.newaxis] *  dJKeM[:,np.newaxis,:,np.newaxis,np.newaxis] 

    eIIbasisuse = invKJKeM[:,np.newaxis,np.newaxis,np.newaxis,:]*np.conj(invKJKeM[np.newaxis,:,np.newaxis,np.newaxis,:]) * MJKeM[:,np.newaxis,np.newaxis,:,np.newaxis] * np.conj(MJKeM[np.newaxis,:,np.newaxis,:,np.newaxis]) * invdJKeM[:,np.newaxis,:,np.newaxis,np.newaxis] * invdJKeM[np.newaxis,:,:,np.newaxis,np.newaxis]

    eEIbasisuse = KJKeM[:,np.newaxis,np.newaxis,np.newaxis,:]*np.conj(invKJKeM[np.newaxis,:,np.newaxis,np.newaxis,:]) * MJKeM[:,np.newaxis,np.newaxis,:,np.newaxis] * np.conj(MJKeM[np.newaxis,:,np.newaxis,:,np.newaxis]) * dJKeM[:,np.newaxis,:,np.newaxis,np.newaxis] * invdJKeM[np.newaxis,:,:,np.newaxis,np.newaxis]

    eIEbasisuse = invKJKeM[:,np.newaxis,np.newaxis,np.newaxis,:]*np.conj(KJKeM[np.newaxis,:,np.newaxis,np.newaxis,:]) * MJKeM[:,np.newaxis,np.newaxis,:,np.newaxis] * np.conj(MJKeM[np.newaxis,:,np.newaxis,:,np.newaxis]) * invdJKeM[:,np.newaxis,:,np.newaxis,np.newaxis] * dJKeM[np.newaxis,:,:,np.newaxis,np.newaxis]


    oEEbasisuse = KJKoM[:,np.newaxis,np.newaxis, np.newaxis,:]*np.conj(KJKoM[np.newaxis,:,np.newaxis,np.newaxis,:]) * MJKoM[:,np.newaxis,np.newaxis,:,np.newaxis] * np.conj(MJKoM[np.newaxis,:,np.newaxis,:,np.newaxis]) * dJKoM[:,np.newaxis,:,np.newaxis,np.newaxis] * dJKoM[np.newaxis,:,:,np.newaxis,np.newaxis]

    oIIbasisuse = invKJKoM[:,np.newaxis,np.newaxis, np.newaxis,:]*np.conj(invKJKoM[np.newaxis,:,np.newaxis,np.newaxis,:]) * MJKoM[:,np.newaxis,np.newaxis,:,np.newaxis] * np.conj(MJKoM[np.newaxis,:,np.newaxis,:,np.newaxis]) * invdJKoM[:,np.newaxis,:,np.newaxis,np.newaxis] * invdJKoM[np.newaxis,:,:,np.newaxis,np.newaxis]

    oEIbasisuse = KJKoM[:,np.newaxis,np.newaxis, np.newaxis,:]*np.conj(invKJKoM[np.newaxis,:,np.newaxis,np.newaxis,:]) * MJKoM[:,np.newaxis,np.newaxis,:,np.newaxis] * np.conj(MJKoM[np.newaxis,:,np.newaxis,:,np.newaxis]) * dJKoM[:,np.newaxis,:,np.newaxis,np.newaxis] * invdJKoM[np.newaxis,:,:,np.newaxis,np.newaxis]

    oIEbasisuse = invKJKoM[:,np.newaxis,np.newaxis, np.newaxis,:]*np.conj(KJKoM[np.newaxis,:,np.newaxis,np.newaxis,:]) * MJKoM[:,np.newaxis,np.newaxis,:,np.newaxis] * np.conj(MJKoM[np.newaxis,:,np.newaxis,:,np.newaxis]) * invdJKoM[:,np.newaxis,:,np.newaxis,np.newaxis] * dJKoM[np.newaxis,:,:,np.newaxis,np.newaxis]



    print("V SIZE: ", v6d.shape)
    print("KJKeM: ", KJKeM.shape)
    print("MJKeM: ", MJKeM.shape)
    print("dJKeM: ", dJKeM.shape)
    print("PSInlm: ", PSInlm.shape)

    sys.stdout.flush()
    eEEebasisuse = np.reshape(eEEbasisuse,(shortJKeM*shortJKeM,angleNum*angleNum*thetaNum),order='C')
    eIIebasisuse = np.reshape(eIIbasisuse,(shortJKeM*shortJKeM,angleNum*angleNum*thetaNum),order='C')
    eEIebasisuse = np.reshape(eEIbasisuse,(shortJKeM*shortJKeM,angleNum*angleNum*thetaNum),order='C')
    eIEebasisuse = np.reshape(eIEbasisuse,(shortJKeM*shortJKeM,angleNum*angleNum*thetaNum),order='C')

    oEEebasisuse = np.reshape(oEEbasisuse,(shortJKoM*shortJKoM,angleNum*angleNum*thetaNum),order='C')
    oIIebasisuse = np.reshape(oIIbasisuse,(shortJKoM*shortJKoM,angleNum*angleNum*thetaNum),order='C')
    oEIebasisuse = np.reshape(oEIbasisuse,(shortJKoM*shortJKoM,angleNum*angleNum*thetaNum),order='C')
    oIEebasisuse = np.reshape(oIEbasisuse,(shortJKoM*shortJKoM,angleNum*angleNum*thetaNum),order='C')


    v6d = np.reshape(v6d, (cageGridNum*cageAngleNum*cageAngleNum, thetaNum*angleNum*angleNum), order='C')  #


    print("eEE i")
    tempa = np.tensordot(eEEebasisuse, v6d, axes=([1],[1]))
    tempa = np.reshape(tempa,(shortJKeM,shortJKeM,cageGridNum*cageAngleNum*cageAngleNum),order='C')
    vNJKeMEE = np.tensordot(tempa,(PSInlm[:,np.newaxis,:]*PSIconjnlm[np.newaxis,:,:]), axes=([2],[2]))

 
    sys.stdout.flush()
    print("eII i")
    tempa = np.tensordot(eIIebasisuse, v6d, axes=([1],[1]))
    tempa = np.reshape(tempa,(shortJKeM,shortJKeM,cageGridNum*cageAngleNum*cageAngleNum),order='C')
    vNJKeMII = np.tensordot(tempa, (PSInlminv[:,np.newaxis,:]*PSIconjnlminv[np.newaxis,:,:]), axes=([2],[2]))

    sys.stdout.flush()

    print("eEI i")
    tempa = np.tensordot(eEIebasisuse, v6d, axes=([1],[1]))
    tempa = np.reshape(tempa,(shortJKeM,shortJKeM,cageGridNum*cageAngleNum*cageAngleNum),order='C')
    vNJKeMEI = np.tensordot(tempa, (PSInlm[:,np.newaxis,:]*PSIconjnlminv[np.newaxis,:,:]), axes=([2],[2]))
    
    sys.stdout.flush()
    print("eIE i")
    tempa = np.tensordot(eIEebasisuse, v6d, axes=([1],[1]))
    tempa = np.reshape(tempa,(shortJKeM,shortJKeM,cageGridNum*cageAngleNum*cageAngleNum),order='C')
    vNJKeMIE = np.tensordot(tempa, (PSInlminv[:,np.newaxis,:]*PSIconjnlm[np.newaxis,:,:]), axes=([2],[2]))
 
    sys.stdout.flush()
    print("oEE i")
    tempa = np.tensordot(oEEebasisuse, v6d, axes=([1],[1]))
    tempa = np.reshape(tempa,(shortJKoM,shortJKoM,cageGridNum*cageAngleNum*cageAngleNum),order='C')
    vNJKoMEE = np.tensordot(tempa, (PSInlm[:,np.newaxis,:]*PSIconjnlm[np.newaxis,:,:]), axes=([2],[2]))
 
    sys.stdout.flush()
    print("oII i")
    tempa = np.tensordot(oIIebasisuse, v6d, axes=([1],[1]))
    tempa = np.reshape(tempa,(shortJKoM,shortJKoM,cageGridNum*cageAngleNum*cageAngleNum),order='C')
    vNJKoMII = np.tensordot(tempa, (PSInlminv[:,np.newaxis,:]*PSIconjnlminv[np.newaxis,:,:]), axes=([2],[2]))
    
    sys.stdout.flush()
    print("oEI i")
    tempa = np.tensordot(oEIebasisuse, v6d, axes=([1],[1]))
    tempa = np.reshape(tempa,(shortJKoM,shortJKoM,cageGridNum*cageAngleNum*cageAngleNum),order='C')
    vNJKoMEI = np.tensordot(tempa, (PSInlm[:,np.newaxis,:]*PSIconjnlminv[np.newaxis,:,:]), axes=([2],[2]))

    sys.stdout.flush()
    print("oIE i")
    tempa = np.tensordot(oIEebasisuse, v6d, axes=([1],[1]))
    tempa = np.reshape(tempa,(shortJKoM,shortJKoM,cageGridNum*cageAngleNum*cageAngleNum),order='C')
    vNJKoMIE = np.tensordot(tempa, (PSInlminv[:,np.newaxis,:]*PSIconjnlm[np.newaxis,:,:]), axes=([2],[2]))

    
    print("hmmmMMMMMMYES")
    
    sys.stdout.flush()
    
    agSortStart = datetime.now()
    for s in range(Agcounter):
        for y in range(Agcounter):
            vAg[s,y] += vNJKeMEE[shortJKeMreverse[AgQuantumNumList[s,3],AgQuantumNumList[s,4],AgQuantumNumList[s,5]], shortJKeMreverse[AgQuantumNumList[y,3],AgQuantumNumList[y,4],AgQuantumNumList[y,5]], Nnlmreverse[AgQuantumNumList[s,0],AgQuantumNumList[s,1],AgQuantumNumList[s,2]], Nnlmreverse[AgQuantumNumList[y,0],AgQuantumNumList[y,1],AgQuantumNumList[y,2]]] 
            vAg[s,y] += vNJKeMII[shortJKeMreverse[AgQuantumNumList[s,3],AgQuantumNumList[s,4],AgQuantumNumList[s,5]], shortJKeMreverse[AgQuantumNumList[y,3],AgQuantumNumList[y,4],AgQuantumNumList[y,5]], Nnlmreverse[AgQuantumNumList[s,0],AgQuantumNumList[s,1],AgQuantumNumList[s,2]], Nnlmreverse[AgQuantumNumList[y,0],AgQuantumNumList[y,1],AgQuantumNumList[y,2]]] 
            vAg[s,y] += vNJKeMEI[shortJKeMreverse[AgQuantumNumList[s,3],AgQuantumNumList[s,4],AgQuantumNumList[s,5]], shortJKeMreverse[AgQuantumNumList[y,3],AgQuantumNumList[y,4],AgQuantumNumList[y,5]], Nnlmreverse[AgQuantumNumList[s,0],AgQuantumNumList[s,1],AgQuantumNumList[s,2]], Nnlmreverse[AgQuantumNumList[y,0],AgQuantumNumList[y,1],AgQuantumNumList[y,2]]] 
            vAg[s,y] += vNJKeMIE[shortJKeMreverse[AgQuantumNumList[s,3],AgQuantumNumList[s,4],AgQuantumNumList[s,5]], shortJKeMreverse[AgQuantumNumList[y,3],AgQuantumNumList[y,4],AgQuantumNumList[y,5]], Nnlmreverse[AgQuantumNumList[s,0],AgQuantumNumList[s,1],AgQuantumNumList[s,2]], Nnlmreverse[AgQuantumNumList[y,0],AgQuantumNumList[y,1],AgQuantumNumList[y,2]]] 
    agSortEnd = datetime.now()
    
    print("###############################################")
    print("###############################################")
    print("###############################################")
    print("###############################################")
    print(vAg.shape, np.sum(vAg), np.sum(np.diag(vAg))),  np.sum(vAg - np.diag(np.diag(vAg)))
    print("###############################################")
    print("###############################################")
    print("###############################################")
    
    sys.stdout.flush()
    auSortStart = datetime.now()
    for s in range(Aucounter):
        for y in range(Aucounter):

            vAu[s,y] += vNJKeMEE[shortJKeMreverse[AuQuantumNumList[s,3],AuQuantumNumList[s,4],AuQuantumNumList[s,5]], shortJKeMreverse[AuQuantumNumList[y,3],AuQuantumNumList[y,4],AuQuantumNumList[y,5]], Nnlmreverse[AuQuantumNumList[s,0],AuQuantumNumList[s,1],AuQuantumNumList[s,2]], Nnlmreverse[AuQuantumNumList[y,0],AuQuantumNumList[y,1],AuQuantumNumList[y,2]]] 

            vAu[s,y] += vNJKeMII[shortJKeMreverse[AuQuantumNumList[s,3],AuQuantumNumList[s,4],AuQuantumNumList[s,5]], shortJKeMreverse[AuQuantumNumList[y,3],AuQuantumNumList[y,4],AuQuantumNumList[y,5]], Nnlmreverse[AuQuantumNumList[s,0],AuQuantumNumList[s,1],AuQuantumNumList[s,2]], Nnlmreverse[AuQuantumNumList[y,0],AuQuantumNumList[y,1],AuQuantumNumList[y,2]]] 

            vAu[s,y] += (-1.0)*vNJKeMEI[shortJKeMreverse[AuQuantumNumList[s,3],AuQuantumNumList[s,4],AuQuantumNumList[s,5]], shortJKeMreverse[AuQuantumNumList[y,3],AuQuantumNumList[y,4],AuQuantumNumList[y,5]], Nnlmreverse[AuQuantumNumList[s,0],AuQuantumNumList[s,1],AuQuantumNumList[s,2]], Nnlmreverse[AuQuantumNumList[y,0],AuQuantumNumList[y,1],AuQuantumNumList[y,2]]] 

            vAu[s,y] += (-1.0)*vNJKeMIE[shortJKeMreverse[AuQuantumNumList[s,3],AuQuantumNumList[s,4],AuQuantumNumList[s,5]], shortJKeMreverse[AuQuantumNumList[y,3],AuQuantumNumList[y,4],AuQuantumNumList[y,5]], Nnlmreverse[AuQuantumNumList[s,0],AuQuantumNumList[s,1],AuQuantumNumList[s,2]], Nnlmreverse[AuQuantumNumList[y,0],AuQuantumNumList[y,1],AuQuantumNumList[y,2]]] 
    auSortEnd = datetime.now()
    print(vAu.shape, np.sum(vAu), np.sum(np.diag(vAu))),  np.sum(vAu - np.diag(np.diag(vAu)))
    
    bgSortStart = datetime.now()
    for s in range(Bgcounter):
        for y in range(Bgcounter):
            vBg[s,y] += vNJKoMEE[shortJKoMreverse[BgQuantumNumList[s,3],BgQuantumNumList[s,4],BgQuantumNumList[s,5]], shortJKoMreverse[BgQuantumNumList[y,3],BgQuantumNumList[y,4],BgQuantumNumList[y,5]], Nnlmreverse[BgQuantumNumList[s,0],BgQuantumNumList[s,1],BgQuantumNumList[s,2]], Nnlmreverse[BgQuantumNumList[y,0],BgQuantumNumList[y,1],BgQuantumNumList[y,2]]] 
            vBg[s,y] += vNJKoMII[shortJKoMreverse[BgQuantumNumList[s,3],BgQuantumNumList[s,4],BgQuantumNumList[s,5]], shortJKoMreverse[BgQuantumNumList[y,3],BgQuantumNumList[y,4],BgQuantumNumList[y,5]], Nnlmreverse[BgQuantumNumList[s,0],BgQuantumNumList[s,1],BgQuantumNumList[s,2]], Nnlmreverse[BgQuantumNumList[y,0],BgQuantumNumList[y,1],BgQuantumNumList[y,2]]] 
            vBg[s,y] += vNJKoMEI[shortJKoMreverse[BgQuantumNumList[s,3],BgQuantumNumList[s,4],BgQuantumNumList[s,5]], shortJKoMreverse[BgQuantumNumList[y,3],BgQuantumNumList[y,4],BgQuantumNumList[y,5]], Nnlmreverse[BgQuantumNumList[s,0],BgQuantumNumList[s,1],BgQuantumNumList[s,2]], Nnlmreverse[BgQuantumNumList[y,0],BgQuantumNumList[y,1],BgQuantumNumList[y,2]]] 
            vBg[s,y] += vNJKoMIE[shortJKoMreverse[BgQuantumNumList[s,3],BgQuantumNumList[s,4],BgQuantumNumList[s,5]], shortJKoMreverse[BgQuantumNumList[y,3],BgQuantumNumList[y,4],BgQuantumNumList[y,5]], Nnlmreverse[BgQuantumNumList[s,0],BgQuantumNumList[s,1],BgQuantumNumList[s,2]], Nnlmreverse[BgQuantumNumList[y,0],BgQuantumNumList[y,1],BgQuantumNumList[y,2]]] 
    bgSortEnd = datetime.now()
    
    print(vBg.shape, np.sum(vBg), np.sum(np.diag(vBg))),  np.sum(vBg - np.diag(np.diag(vBg)))
    buSortStart = datetime.now()
    for s in range(Bucounter):
        for y in range(Bucounter):
            vBu[s,y] += vNJKoMEE[shortJKoMreverse[BuQuantumNumList[s,3],BuQuantumNumList[s,4],BuQuantumNumList[s,5]], shortJKoMreverse[BuQuantumNumList[y,3],BuQuantumNumList[y,4],BuQuantumNumList[y,5]], Nnlmreverse[BuQuantumNumList[s,0],BuQuantumNumList[s,1],BuQuantumNumList[s,2]], Nnlmreverse[BuQuantumNumList[y,0],BuQuantumNumList[y,1],BuQuantumNumList[y,2]]] 
            vBu[s,y] += vNJKoMII[shortJKoMreverse[BuQuantumNumList[s,3],BuQuantumNumList[s,4],BuQuantumNumList[s,5]], shortJKoMreverse[BuQuantumNumList[y,3],BuQuantumNumList[y,4],BuQuantumNumList[y,5]], Nnlmreverse[BuQuantumNumList[s,0],BuQuantumNumList[s,1],BuQuantumNumList[s,2]], Nnlmreverse[BuQuantumNumList[y,0],BuQuantumNumList[y,1],BuQuantumNumList[y,2]]] 
            vBu[s,y] += (-1.0)*vNJKoMIE[shortJKoMreverse[BuQuantumNumList[s,3],BuQuantumNumList[s,4],BuQuantumNumList[s,5]], shortJKoMreverse[BuQuantumNumList[y,3],BuQuantumNumList[y,4],BuQuantumNumList[y,5]], Nnlmreverse[BuQuantumNumList[s,0],BuQuantumNumList[s,1],BuQuantumNumList[s,2]], Nnlmreverse[BuQuantumNumList[y,0],BuQuantumNumList[y,1],BuQuantumNumList[y,2]]] 
            vBu[s,y] += (-1.0)*vNJKoMEI[shortJKoMreverse[BuQuantumNumList[s,3],BuQuantumNumList[s,4],BuQuantumNumList[s,5]], shortJKoMreverse[BuQuantumNumList[y,3],BuQuantumNumList[y,4],BuQuantumNumList[y,5]], Nnlmreverse[BuQuantumNumList[s,0],BuQuantumNumList[s,1],BuQuantumNumList[s,2]], Nnlmreverse[BuQuantumNumList[y,0],BuQuantumNumList[y,1],BuQuantumNumList[y,2]]] 
    buSortEnd = datetime.now()
    
    print(vBu.shape, np.sum(vBu), np.sum(np.diag(vBu))),  np.sum(vBu - np.diag(np.diag(vBu)))
    
    #############################################################################################################
    #############################################################################################################
    #############################################################################################################
    #############################################################################################################
    #############################################################################################################
    print("OKAY MADE IT UP TO HERE")
    
    sys.stdout.flush()
    HAg += vAg
    HAu += vAu
    HBg += vBg
    HBu += vBu
    
    
    #########################################################################
    ## calculate Hrot (Asymmetric Top Hamiltonian in Symmetric Top Basis)  ##
    #########################################################################
    HrotKe = np.zeros((JKeM,JKeM),dtype=float)
    HrotKo = np.zeros((JKoM,JKoM),dtype=float)
    
    
    
    
    for jkm in range(JKeM):
        for jkmp in range(JKeM):
            if JKeMQuantumNumList[jkm,0]==JKeMQuantumNumList[jkmp,0] and JKeMQuantumNumList[jkm,2]==JKeMQuantumNumList[jkmp,2]:
                if JKeMQuantumNumList[jkm,1]==(JKeMQuantumNumList[jkmp,1]-2):
                    HrotKe[jkm,jkmp] += 0.25*(Ah2o-Ch2o)*off_diag(JKeMQuantumNumList[jkm,0],JKeMQuantumNumList[jkm,1])*off_diag(JKeMQuantumNumList[jkm,0],JKeMQuantumNumList[jkm,1]+1)
                elif JKeMQuantumNumList[jkm,1]==(JKeMQuantumNumList[jkmp,1]+2):
                    HrotKe[jkm,jkmp] += 0.25*(Ah2o-Ch2o)*off_diag(JKeMQuantumNumList[jkm,0],JKeMQuantumNumList[jkm,1]-1)*off_diag(JKeMQuantumNumList[jkm,0],JKeMQuantumNumList[jkm,1]-2)
                elif JKeMQuantumNumList[jkm,1]==(JKeMQuantumNumList[jkmp,1]):
                    HrotKe[jkm,jkmp] += (0.5*(Ah2o + Ch2o)*(JKeMQuantumNumList[jkm,0]*(JKeMQuantumNumList[jkm,0]+1)) + (Bh2o - 0.5*(Ah2o+Ch2o)) * ((JKeMQuantumNumList[jkm,1])**2))
    
    
    for jkm in range(JKoM):
        for jkmp in range(JKoM):
            if JKoMQuantumNumList[jkm,0]==JKoMQuantumNumList[jkmp,0] and JKoMQuantumNumList[jkm,2]==JKoMQuantumNumList[jkmp,2]:
                if JKoMQuantumNumList[jkm,1]==(JKoMQuantumNumList[jkmp,1]-2):
                    HrotKo[jkm,jkmp] += 0.25*(Ah2o-Ch2o)*off_diag(JKoMQuantumNumList[jkm,0],JKoMQuantumNumList[jkm,1])*off_diag(JKoMQuantumNumList[jkm,0],JKoMQuantumNumList[jkm,1]+1)
                elif JKoMQuantumNumList[jkm,1]==(JKoMQuantumNumList[jkmp,1]+2):
                    HrotKo[jkm,jkmp] += 0.25*(Ah2o-Ch2o)*off_diag(JKoMQuantumNumList[jkm,0],JKoMQuantumNumList[jkm,1]-1)*off_diag(JKoMQuantumNumList[jkm,0],JKoMQuantumNumList[jkm,1]-2)
                elif JKoMQuantumNumList[jkm,1]==(JKoMQuantumNumList[jkmp,1]):
                    HrotKo[jkm,jkmp] += (0.5*(Ah2o + Ch2o)*(JKoMQuantumNumList[jkm,0]*(JKoMQuantumNumList[jkm,0]+1)) + (Bh2o - 0.5*(Ah2o+Ch2o)) * ((JKoMQuantumNumList[jkm,1])**2))
    
    
    ##################################################################################################################################################
    rotest = LA.eigh(HrotKo)[0] # 
    azdx = rotest.argsort()     #   prints out eigenvalues for pure asymmetric top rotor (z_ORTHOz)
    rotest = rotest[azdx]       #
    print(rotest)                # 
    ###########################################################################################
    
    ##################################################################################################################################################
    rotest = LA.eigh(HrotKe)[0] # 
    azdx = rotest.argsort()     #   prints out eigenvalues for pure asymmetric top rotor (z_ORTHOz)
    rotest = rotest[azdx]       #
    print(rotest)                # 
    ###########################################################################################
    
    
    #####################################################################################################
    rsqElement = np.zeros((Nnlm,Nnlm),float)
    
    for n in range(Nnlm):
        for n2 in range(Nnlm):
            k = int((NnlmQuantumNumList[n,0] - NnlmQuantumNumList[n,1] ) / 2)
            k2 = int((NnlmQuantumNumList[n2,0] - NnlmQuantumNumList[n2,1] ) / 2)
            tau = k - k2
    
            if NnlmQuantumNumList[n,1] == NnlmQuantumNumList[n2,1] and NnlmQuantumNumList[n,2] == NnlmQuantumNumList[n2,2]:# and np.abs(tau) <=1.:
    
                m = NnlmQuantumNumList[n,1] + 3./2.
                rsqTempA = Nkl[k,NnlmQuantumNumList[n,1]] * Nkl[k2,NnlmQuantumNumList[n2,1]] * ((-1.)**(k+k2+1))/((2*nu)**(m))
    
                if k >= k2:
    
                    rsqTempB = 0.0
    
                    for sig in range(k2+1):
                        rsqTempB += binom(1,tau+sig)*binom(1,sig)*np.math.gamma(m + k2 - sig +1) / np.math.factorial(k2-sig)
    
                    rsqElement[n,n2] += rsqTempA * rsqTempB 
    
                else:
                    k_use = int((NnlmQuantumNumList[n2,0] - NnlmQuantumNumList[n2,1] ) /2) 
                    k2_use = int((NnlmQuantumNumList[n,0] - NnlmQuantumNumList[n,1] ) /2) 
                    tau_use = k_use - k2_use

                    m_use = NnlmQuantumNumList[n,1] + 3. /2.
                    rsqTempA = Nkl[k_use,NnlmQuantumNumList[n2,1]] * Nkl[k2_use,NnlmQuantumNumList[n,1]] * ((-1.)**(k_use+k2_use+1))/((2*nu)**(m))
                    rsqTempB = 0.0
    
                    for sig in range(k2_use+1):
                        rsqTempB += binom(1,tau_use+sig)*binom(1,sig)*np.math.gamma(m + k2_use - sig +1) / np.math.factorial(k2_use-sig)
    
                    rsqElement[n,n2] += rsqTempA *rsqTempB 
                  
    rsqElement *= (0.25*hbar*omega)/jpcm
    
    sys.stdout.flush()
    print("rsqELEMENT MIN MAX")
    print("rsqELEMENT MIN MAX")
    
    print("min: ", np.min(rsqElement))
    print("max: ", np.max(rsqElement))
    print("min abs ", np.min(np.abs(rsqElement)))
    print(np.sum(rsqElement - (rsqElement.T)))
    #####################################################################################################
    #nqsqnp = np.zeros((Nmax+1,Nmax+1),float)                                                           #
    ##for n in range(Nmax+1):                                                                           #
    #    for n2 in range(Nmax+1):                                                                       #
    #        for q in range(gridNum):                                                                   #
    #            nqsqnp[n,n2] += xyzvec[n,q]*xyzvec[n2,q]*(xyzgrid[q]**2)/(1.*(10**9)**2)               #
    #                                                                                                   #
    #nqsqnp *= -(0.5*massh2o*omega*omega) / jpcm                                                        #
    #####################################################################################################
    






    
    auTRS = datetime.now()
    ###########################################################################################
    # ADD DIAGONAL TR ENERGY Au BLOCK
    tempMat = np.zeros((Aucounter,Aucounter),float)
    
    for i in range(Aucounter):
        if AuQuantumNumList[i,4] == 0:
            Nk = 1./4.
        else:
            Nk = 1./2.
        for j in range(Aucounter):
            if AuQuantumNumList[i,0] == AuQuantumNumList[j,0] and  AuQuantumNumList[i,1] == AuQuantumNumList[j,1] and AuQuantumNumList[i,2] == AuQuantumNumList[j,2] and AuQuantumNumList[i,3] == AuQuantumNumList[j,3] and AuQuantumNumList[i,4] == AuQuantumNumList[j,4] and  AuQuantumNumList[i,5] == AuQuantumNumList[j,5]:
                temp = 1.
                if AuQuantumNumList[i,4]==0:
                    temp += (-1.)**(AuQuantumNumList[i,1] + AuQuantumNumList[i,3]  + 1.0)
                tempMat[i,j] += (AuQuantumNumList[i,0] + 3./2.)*temp*Nk*2.0
    
    tempMat *= hbar*omega/jpcm
    HAu += tempMat
    ###########################################################################################
    auTRE = datetime.now()









    
    agTRS = datetime.now()
    ###########################################################################################
    # ADD DIAGONAL TR ENERGY Ag BLOCK
    tempMat = np.zeros((Agcounter,Agcounter),float)
    for i in range(Agcounter):
        if AgQuantumNumList[i,4] == 0:
            Nk = 1./4.
        else:
            Nk = 1./2.
        for j in range(Agcounter):
            if AgQuantumNumList[i,0] == AgQuantumNumList[j,0] and  AgQuantumNumList[i,1] == AgQuantumNumList[j,1] and AgQuantumNumList[i,2] == AgQuantumNumList[j,2] and AgQuantumNumList[i,3] == AgQuantumNumList[j,3] and AgQuantumNumList[i,4] == AgQuantumNumList[j,4] and  AgQuantumNumList[i,5] == AgQuantumNumList[j,5]:
                temp = 1.
                if AgQuantumNumList[i,4]==0:
                    temp += (-1.)**(AgQuantumNumList[i,1] + AgQuantumNumList[i,3] + AgQuantumNumList[i,4])
                tempMat[i,j] += (AgQuantumNumList[i,0] + 3./2.)*temp*Nk*2.0
    tempMat *= hbar*omega/jpcm
    HAg += tempMat
    ###########################################################################################
    agTRE = datetime.now()









    
    buTRS = datetime.now()
    ###########################################################################################
     # ADD DIAGONAL TR ENERGY Bu BLOCK
    tempMat = np.zeros((Bucounter,Bucounter),float)
    for i in range(Bucounter):
        for j in range(Bucounter):
            if BuQuantumNumList[i,0] == BuQuantumNumList[j,0] and  BuQuantumNumList[i,1] == BuQuantumNumList[j,1] and BuQuantumNumList[i,2] == BuQuantumNumList[j,2] and BuQuantumNumList[i,3] == BuQuantumNumList[j,3] and BuQuantumNumList[i,4] == BuQuantumNumList[j,4] and  BuQuantumNumList[i,5] == BuQuantumNumList[j,5]:
                tempMat[i,j] += (BuQuantumNumList[i,0] + 3./2.)
    tempMat *= hbar*omega/jpcm
    HBu += tempMat          
    ###########################################################################################
    buTRE = datetime.now()
   








 
    bgTRS = datetime.now()
    ###########################################################################################
    # ADD DIAGONAL TR ENERGY Bg BLOCK
    tempMat = np.zeros((Bgcounter,Bgcounter),float)
    for i in range(Bgcounter):
        for j in range(Bgcounter):
            if BgQuantumNumList[i,0] == BgQuantumNumList[j,0] and  BgQuantumNumList[i,1] == BgQuantumNumList[j,1] and BgQuantumNumList[i,2] == BgQuantumNumList[j,2] and BgQuantumNumList[i,3] == BgQuantumNumList[j,3] and BgQuantumNumList[i,4] == BgQuantumNumList[j,4] and  BgQuantumNumList[i,5] == BgQuantumNumList[j,5]:
                tempMat[i,j] += (BgQuantumNumList[i,0] + 3./2.)
    tempMat *= hbar*omega/jpcm
    HBg += tempMat
    ###########################################################################################
    bgTRE = datetime.now()




    
    auqS = datetime.now()
    ###########################################################################################
    # ADD -0.5mw^2Q^2 term Au BLOCK
    tempMat = np.zeros((Aucounter,Aucounter),float)
    for i in range(Aucounter):
        if AuQuantumNumList[i,4]==0:
            Nk = 1./4.
        else:
            Nk = 1./2.
        for j in range(Aucounter):
    
            if AuQuantumNumList[i,1]==AuQuantumNumList[j,1] and AuQuantumNumList[i,2]==AuQuantumNumList[j,2] and AuQuantumNumList[i,3] == AuQuantumNumList[j,3] and AuQuantumNumList[i,4] == AuQuantumNumList[j,4] and  AuQuantumNumList[i,5] == AuQuantumNumList[j,5]:
                temp = 1.
                if AuQuantumNumList[i,4] == 0:
                    temp += ((-1.)**(AuQuantumNumList[i,1] + AuQuantumNumList[i,3] + 1))
                tempMat[i,j] += 2.0*Nk*temp*rsqElement[Nnlmreverse[AuQuantumNumList[i,0],AuQuantumNumList[i,1],AuQuantumNumList[i,2]],Nnlmreverse[AuQuantumNumList[j,0],AuQuantumNumList[j,1],AuQuantumNumList[j,2]]]
    
    HAu += tempMat
    ###########################################################################################
    auqE = datetime.now()





    
    agqS = datetime.now()
    ###########################################################################################
    # ADD -0.5mw^2Q^2 term Ag BLOCK
    tempMat = np.zeros((Agcounter,Agcounter),float)
    
    for i in range(Agcounter):
        if AgQuantumNumList[i,4]==0:
            Nk = 1./4.
        else:
            Nk = 1./2.
        for j in range(Agcounter):
            if AgQuantumNumList[i,1]==AgQuantumNumList[j,1] and AgQuantumNumList[i,2]==AgQuantumNumList[j,2] and AgQuantumNumList[i,3] == AgQuantumNumList[j,3] and AgQuantumNumList[i,4] == AgQuantumNumList[j,4] and  AgQuantumNumList[i,5] == AgQuantumNumList[j,5]:
                temp = 1.
                if AgQuantumNumList[i,4] == 0:
                    temp += ((-1.)**(AgQuantumNumList[i,1] +  AgQuantumNumList[i,3]))
                tempMat[i,j] += 2.0*Nk*temp*rsqElement[Nnlmreverse[AgQuantumNumList[i,0],AgQuantumNumList[i,1],AgQuantumNumList[i,2]],Nnlmreverse[AgQuantumNumList[j,0],AgQuantumNumList[j,1],AgQuantumNumList[j,2]]]
    
    HAg += tempMat
    ###########################################################################################
    agqE = datetime.now()
   







 
    
    buqS = datetime.now()
    ###########################################################################################
    # ADD -0.5mw^2Q^2 term Bu BLOCK
    tempMat = np.zeros((Bucounter,Bucounter),float)
    
    for i in range(Bucounter):
        for j in range(Bucounter):
            if BuQuantumNumList[i,1]==BuQuantumNumList[j,1] and BuQuantumNumList[i,2]==BuQuantumNumList[j,2] and BuQuantumNumList[i,3] == BuQuantumNumList[j,3] and BuQuantumNumList[i,4] == BuQuantumNumList[j,4] and  BuQuantumNumList[i,5] == BuQuantumNumList[j,5]:
                tempMat[i,j] += rsqElement[Nnlmreverse[BuQuantumNumList[i,0],BuQuantumNumList[i,1],BuQuantumNumList[i,2]],Nnlmreverse[BuQuantumNumList[j,0],BuQuantumNumList[j,1],BuQuantumNumList[j,2]]]
    
    HBu += tempMat
    ###########################################################################################
    buqE = datetime.now()








    
    bgqS = datetime.now()
    ###########################################################################################
    # ADD -0.5mw^2Q^2 term Bg BLOCK
    tempMat = np.zeros((Bgcounter,Bgcounter),float)
    
    for i in range(Bgcounter):
        for j in range(Bgcounter):
            if BgQuantumNumList[i,1] ==BgQuantumNumList[j,1] and BgQuantumNumList[i,2]==BgQuantumNumList[j,2] and BgQuantumNumList[i,3] == BgQuantumNumList[j,3] and BgQuantumNumList[i,4] == BgQuantumNumList[j,4] and  BgQuantumNumList[i,5] == BgQuantumNumList[j,5]:
                tempMat[i,j] += rsqElement[Nnlmreverse[BgQuantumNumList[i,0],BgQuantumNumList[i,1],BgQuantumNumList[i,2]],Nnlmreverse[BgQuantumNumList[j,0],BgQuantumNumList[j,1],BgQuantumNumList[j,2]]]
                    
    HBg += tempMat
    ###########################################################################################
    bgqE = datetime.now()






    
    auRS = datetime.now()
    ###########################################################################################
    # ADD Hrot to  Au BLOCK
    tempMat = np.zeros((Aucounter,Aucounter),float)
    
    for i in range(Aucounter):
        if AuQuantumNumList[i,4]==0:
            Nki = 1./2.
        else:
            Nki = 1./(np.sqrt(2.))
        for j in range(Aucounter):
            if AuQuantumNumList[j,4]==0:
                Nkj = 1./2.
            else:
                Nkj = 1./(np.sqrt(2.))
            temp = 0.
            if AuQuantumNumList[i,0] == AuQuantumNumList[j,0] and AuQuantumNumList[i,1] == AuQuantumNumList[j,1] and  AuQuantumNumList[i,2] == AuQuantumNumList[j,2]:
                tempMat[i,j] += HrotKe[JKeMreverse[AuQuantumNumList[i,3],AuQuantumNumList[i,4],AuQuantumNumList[i,5]],JKeMreverse[AuQuantumNumList[j,3],AuQuantumNumList[j,4],AuQuantumNumList[j,5]]]
                tempMat[i,j] += HrotKe[JKeMreverse[AuQuantumNumList[i,3],-AuQuantumNumList[i,4],AuQuantumNumList[i,5]],JKeMreverse[AuQuantumNumList[j,3],-AuQuantumNumList[j,4],AuQuantumNumList[j,5]]]
                #tempMat[i,j] += (-1.0)**(AuQuantumNumList[i,4] + AuQuantumNumList[j,4]) * HrotKe[JKeMreverse[AuQuantumNumList[i,3],-AuQuantumNumList[i,4],AuQuantumNumList[i,5]],JKeMreverse[AuQuantumNumList[j,3],-AuQuantumNumList[j,4],AuQuantumNumList[j,5]]]
    
                tempMat[i,j] += (-1.0)**(1 + AuQuantumNumList[j,1] + AuQuantumNumList[j,3] + AuQuantumNumList[j,4]) * HrotKe[JKeMreverse[AuQuantumNumList[i,3],AuQuantumNumList[i,4],AuQuantumNumList[i,5]],JKeMreverse[AuQuantumNumList[j,3],-AuQuantumNumList[j,4],AuQuantumNumList[j,5]]]
                tempMat[i,j] += (-1.0)**(1 + AuQuantumNumList[i,1] + AuQuantumNumList[i,3] + AuQuantumNumList[i,4]) * HrotKe[JKeMreverse[AuQuantumNumList[i,3],-AuQuantumNumList[i,4],AuQuantumNumList[i,5]],JKeMreverse[AuQuantumNumList[j,3],AuQuantumNumList[j,4],AuQuantumNumList[j,5]]]
                tempMat[i,j] *= (Nki*Nkj)
    HAu += tempMat
    ###########################################################################################
    auRE = datetime.now()









    
    agRS = datetime.now()
    ###########################################################################################
    # ADD Hrot to  Ag BLOCK
    tempMat = np.zeros((Agcounter,Agcounter),float)
    
    for i in range(Agcounter):
        if AgQuantumNumList[i,4]==0:
            Nki = 1./2.
        else:
            Nki = 1./(np.sqrt(2.))
        for j in range(Agcounter):
            if AgQuantumNumList[j,4]==0:
                Nkj = 1./2.
            else:
                Nkj = 1./(np.sqrt(2.))
            temp = 0.
            if AgQuantumNumList[i,0] == AgQuantumNumList[j,0] and AgQuantumNumList[i,1] == AgQuantumNumList[j,1] and  AgQuantumNumList[i,2] == AgQuantumNumList[j,2]:
                tempMat[i,j] += HrotKe[JKeMreverse[AgQuantumNumList[i,3],AgQuantumNumList[i,4],AgQuantumNumList[i,5]],JKeMreverse[AgQuantumNumList[j,3],AgQuantumNumList[j,4],AgQuantumNumList[j,5]]]
                tempMat[i,j] += HrotKe[JKeMreverse[AgQuantumNumList[i,3],-AgQuantumNumList[i,4],AgQuantumNumList[i,5]],JKeMreverse[AgQuantumNumList[j,3],-AgQuantumNumList[j,4],AgQuantumNumList[j,5]]]
                #tempMat[i,j] += (-1.0)**(AgQuantumNumList[i,4] + AgQuantumNumList[j,4]) * HrotKe[JKeMreverse[AgQuantumNumList[i,3],-AgQuantumNumList[i,4],AgQuantumNumList[i,5]],JKeMreverse[AgQuantumNumList[j,3],-AgQuantumNumList[j,4],AgQuantumNumList[j,5]]]
    
                tempMat[i,j] += (-1.0)**(AgQuantumNumList[j,1] + AgQuantumNumList[j,3] + AgQuantumNumList[j,4]) * HrotKe[JKeMreverse[AgQuantumNumList[i,3],AgQuantumNumList[i,4],AgQuantumNumList[i,5]],JKeMreverse[AgQuantumNumList[j,3],-AgQuantumNumList[j,4],AgQuantumNumList[j,5]]]
                tempMat[i,j] += (-1.0)**(AgQuantumNumList[i,1] + AgQuantumNumList[i,3] + AgQuantumNumList[i,4]) * HrotKe[JKeMreverse[AgQuantumNumList[i,3],-AgQuantumNumList[i,4],AgQuantumNumList[i,5]],JKeMreverse[AgQuantumNumList[j,3],AgQuantumNumList[j,4],AgQuantumNumList[j,5]]]
    
                tempMat[i,j] *= (Nki*Nkj)
    HAg += tempMat
    ###########################################################################################
    agRE = datetime.now()










    
    buRS = datetime.now()
    ###########################################################################################
    # ADD Hrot to  Bu BLOCK
    tempMat = np.zeros((Bucounter,Bucounter),float)
    
    for i in range(Bucounter):
        for j in range(Bucounter):
            if BuQuantumNumList[i,0] == BuQuantumNumList[j,0] and BuQuantumNumList[i,1] == BuQuantumNumList[j,1] and  BuQuantumNumList[i,2] == BuQuantumNumList[j,2]:
                tempMat[i,j] += HrotKo[JKoMreverse[BuQuantumNumList[i,3],BuQuantumNumList[i,4],BuQuantumNumList[i,5]],JKoMreverse[BuQuantumNumList[j,3],BuQuantumNumList[j,4],BuQuantumNumList[j,5]]]
                tempMat[i,j] += HrotKo[JKoMreverse[BuQuantumNumList[i,3],-BuQuantumNumList[i,4],BuQuantumNumList[i,5]],JKoMreverse[BuQuantumNumList[j,3],-BuQuantumNumList[j,4],BuQuantumNumList[j,5]]]
                tempMat[i,j] += (-1.0)**(1 + BuQuantumNumList[i,1] + BuQuantumNumList[j,3] + BuQuantumNumList[j,4]) * HrotKo[JKoMreverse[BuQuantumNumList[i,3],BuQuantumNumList[i,4],BuQuantumNumList[i,5]],JKoMreverse[BuQuantumNumList[j,3],-BuQuantumNumList[j,4],BuQuantumNumList[j,5]]]
                tempMat[i,j] += (-1.0)**(1 + BuQuantumNumList[i,1] + BuQuantumNumList[i,3] + BuQuantumNumList[i,4]) * HrotKo[JKoMreverse[BuQuantumNumList[i,3],-BuQuantumNumList[i,4],BuQuantumNumList[i,5]],JKoMreverse[BuQuantumNumList[j,3],BuQuantumNumList[j,4],BuQuantumNumList[j,5]]]
    tempMat *= 0.5
    HBu += tempMat
    ###########################################################################################
    buRE = datetime.now()










    
    bgRS = datetime.now()
    ###########################################################################################
    # ADD Hrot to  Bg BLOCK
    tempMat = np.zeros((Bgcounter,Bgcounter),float)
    
    for i in range(Bgcounter):
        for j in range(Bgcounter):
            temp = 0.
            if BgQuantumNumList[i,0] == BgQuantumNumList[j,0] and BgQuantumNumList[i,1] == BgQuantumNumList[j,1] and  BgQuantumNumList[i,2] == BgQuantumNumList[j,2]:
                tempMat[i,j] += HrotKo[JKoMreverse[BgQuantumNumList[i,3],BgQuantumNumList[i,4],BgQuantumNumList[i,5]],JKoMreverse[BgQuantumNumList[j,3],BgQuantumNumList[j,4],BgQuantumNumList[j,5]]]
                tempMat[i,j] += HrotKo[JKoMreverse[BgQuantumNumList[i,3],-BgQuantumNumList[i,4],BgQuantumNumList[i,5]],JKoMreverse[BgQuantumNumList[j,3],-BgQuantumNumList[j,4],BgQuantumNumList[j,5]]]
    
                tempMat[i,j] += (-1.0)**(BgQuantumNumList[i,1] +  BgQuantumNumList[j,3] + BgQuantumNumList[j,4]) * HrotKo[JKoMreverse[BgQuantumNumList[i,3],BgQuantumNumList[i,4],BgQuantumNumList[i,5]],JKoMreverse[BgQuantumNumList[j,3],-BgQuantumNumList[j,4],BgQuantumNumList[j,5]]]
                tempMat[i,j] += (-1.0)**(BgQuantumNumList[i,1] + BgQuantumNumList[i,3] + BgQuantumNumList[i,4]) * HrotKo[JKoMreverse[BgQuantumNumList[i,3],-BgQuantumNumList[i,4],BgQuantumNumList[i,5]],JKoMreverse[BgQuantumNumList[j,3],BgQuantumNumList[j,4],BgQuantumNumList[j,5]]]
    tempMat *= 0.5
    HBg += tempMat
    ###########################################################################################








    bgRE = datetime.now()
    # check to make sure H is hermitian
    print("HAu-HAu.T*", np.sum(HAu - np.conj(HAu.T)))
    print("HAg-HAg.T*", np.sum(HAg - np.conj(HAg.T)))
    print("HBu-HBu.T*", np.sum(HBu - np.conj(HBu.T)))
    print("HBg-HBh.T*", np.sum(HBg - np.conj(HBg.T)))
    #exit()
    #quicksave
    sys.stdout.flush()
    #####################################
    # diagonalize the Hamiltonian and   #
    # sort by eigenvalue                #
    
    HAuS = datetime.now()
    ###########################################
    [AuEVal,AuEVec] = LA.eigh(HAu) 
    Ausort = AuEVal.argsort()
    AuEVal = AuEVal[Ausort]
    AuEVec = AuEVec[:,Ausort]
    print('done ortho u (Au), H calc' )
    HAuE = datetime.now()
    
    sys.stdout.flush()
    HAgS = datetime.now()
    ###########################################
    [AgEVal,AgEVec] = LA.eigh(HAg) 
    Agsort = AgEVal.argsort()
    AgEVal = AgEVal[Agsort]
    AgEVec = AgEVec[:,Agsort]
    print('done ortho g (Ag), H calc')   
    HAgE = datetime.now()
    
    sys.stdout.flush()
    HBuS = datetime.now()         
    ###########################################
    [BuEVal,BuEVec] = LA.eigh(HBu) 
    Busort = BuEVal.argsort()
    BuEVal = BuEVal[Busort]
    BuEVec = BuEVec[:,Busort]
    print('done para u (Bu), H calc')     
    HBuE = datetime.now()
    
    sys.stdout.flush()
    HBgS = datetime.now()       
    ###########################################
    [BgEVal,BgEVec] = LA.eigh(HBg) 
    Bgsort = BgEVal.argsort()
    BgEVal = BgEVal[Bgsort]
    BgEVec = BgEVec[:,Bgsort]
    print('done para g (Bg), H calc') 
    ###########################################
    HBgE = datetime.now()
    sys.stdout.flush()
    
    print(AgEVal[0])
    print(AgEVal-AgEVal[0])
    print(AuEVal-AgEVal[0])
    print(BgEVal - AgEVal[0])
    print(BuEVal - AgEVal[0])
    print(AgEVal[0])
    #exit()
    
    print(np.imag(np.sum(AuEVec[:,0])))
    print(np.real(np.sum(AuEVec[:,0])))
    print(np.sqrt(np.sum(AuEVec[:,0]*np.conj(AuEVec[:,0]))))
    ########################################
    
    endTime = datetime.now()
    ########################################
    
    totalTime = (endTime - startTime)#.seconds
    print(totalTime)
    basisSetup = (basisSetupTime - startTime)#.seconds
    vCalcTime = (vEnd - vStart)#.seconds
    
    sys.stdout.flush()
    #JKeMEEtime = (eEEend - eEEstart)#.seconds
    #votime = JKoMEEtime + JKoMIEtime + JKoMEItime + JKoMIItime
    
    #vtime = vetime+votime
    
    
    agSort = (agSortEnd - agSortStart)#.seconds
    auSort = (auSortEnd - auSortStart)#.seconds
    bgSort = (bgSortEnd - bgSortStart)#.seconds
    buSort = (buSortEnd - buSortStart)#.seconds
    
    sorttime = agSort+auSort+bgSort+buSort
    
    auTR = (auTRE - auTRS)#.seconds
    agTR = (agTRE - agTRS)#.seconds
    bgTR = (bgTRE - bgTRS)#.seconds
    buTR = (buTRE - buTRS)#.seconds
    
    TRtime = auTR + agTR + bgTR + buTR
    
    auq = (auqE - auqS)#.seconds
    agq = (agqE - agqS)#.seconds
    bgq = (bgqE - bgqS)#.seconds
    buq = (buqE - buqS)#.seconds
    
    qtime = auq + agq + buq + bgq
    
    auR = (auRE - auRS)#.seconds
    agR = (agRE - agRS)#.seconds
    buR = (buRE - buRS)#.seconds
    bgR = (bgRE - bgRS)#.seconds
    
    Rtime = auR + agR + buR + bgR
    
    HAgtime = (HAgE - HAgS)#.seconds
    HAutime = (HAuE - HAuS)#.seconds
    HBgtime = (HBgE - HBgS)#.seconds
    HButime = (HBuE - HBuS)#.seconds
    
    Htime = HAgtime + HAutime + HBgtime + HButime
    
    
    
    
    labelA="/warehouse/sbyim/ED_OUTPUT/data/energy/"
    #labelA="data/energy/"
    #labelAii="data/wf/"
    labelAii="/warehouse/sbyim/ED_OUTPUT/data/wf/"
    labelB = str(Jmax)+"_"+str(Nmax)+"_"+str(angleNum)+"_"+str(cageGridNum)+"_"+cage+"_final_sp"
    
    
    #ftotal = totalTime.hour*60*60 + totalTime.minute*60 + totalTime.second + totalTime.microsecond*10**(-6)
    #basetotal = basetotal.hours*60*60 + total.minutes*60 + total.seconds + total.microseconds*10**(-6)
    #vtotal = vtotal.hours*60*60 + vtotal.minutes*60 + vtotal.seconds + vtotal.microseconds*10**(-6)
    #vcalctotal = vcalctotal.hours*60*60 + vcalctotal.minutes*60 + vcalctotal.seconds + vcalctotal.microseconds*10**(-6)
    #sorttotal = sorttotal.hours*60*60 + sorttotal.minutes*60 + sorttotal.seconds + sorttotal.microseconds*10**(-6)
    #TRtotal = TRtotal.hours*60*60 + TRtotal.minutes*60 + TRtotal.seconds + TRtotal.microseconds*10**(-6)
    #Rtotal = Rtotal.hours*60*60 + Rtotal.minutes*60 + Rtotal.seconds + Rtotal.microseconds*10**(-6)
    #Htotal = Htotal.hours*60*60 + Htotal.minutes*60 + Htotal.seconds + Htotal.microseconds*10**(-6)
    
    print(totalTime)
    
    ftotal =  totalTime.seconds + totalTime.microseconds*10**(-6)
    basistotal = basisSetup.seconds + basisSetup.microseconds*10**(-6)
    #vtotal =  vtime.seconds + vtime.microseconds*10**(-6)
    #vcalctotal =  vCalcTime.seconds + vCalcTime.microseconds*10**(-6)
    sorttotal = sorttime.seconds + sorttime.microseconds*10**(-6)
    TRtotal = TRtime.seconds + TRtime.microseconds*10**(-6)
    qtotal = qtime.seconds + qtime.microseconds*10**(-6) 
    Rtotal =  Rtime.seconds + Rtime.microseconds*10**(-6)
    Htotal =  Htime.seconds + Htime.microseconds*10**(-6)
    #print (vcalctotal/ftotal)
    print(ftotal)
    
    profileLabel = "/home/sbyim/workspace/H2O-C60-ED/"+labelB+".time"
    timerOut = open(profileLabel,'w')
    timerOut.write("Total completion time: " + str(totalTime)+" \n")
    #timerOut.write("Basis setup: " +str(basisSetup)+"\n")# micros, "+str((basisSetup)/(totalTime)*100.)+" % total \n")
    #timerOut.write("v calc: " +str(vCalcTime)+"\n")# micros, "+str((vCalcTime)/(totalTime)*100.)+" % total \n")
    #timerOut.write("v mat calc: " +str(vtime)+"\n")# micros, "+str((vtime)/(totalTime)*100.)+" % total \n")
    #timerOut.write("v sort: " +str(sorttime)+"\n")# micros, "+str((sorttime)/(totalTime)*100.)+" % total \n")
    #timerOut.write("TR add: " +str(TRtime)+"\n")# micros, "+str((TRtime)/(totalTime)*100.)+" % total \n")
    #timerOut.write("q add: " +str(qtime)+"\n")# micros, "+str((qtime)/(totalTime)*100.)+" % total \n")
    #timerOut.write("R add: " +str(Rtime)+"\n")# micros, "+str((Rtime)/(totalTime)*100.)+" % total \n")
    #timerOut.write("H diag: " +str(Htime)+"\n")# micros, "+str((Htime)/(totalTime)*100.)+" % total \n")
    timerOut.write("Basis setup: " +str(basisSetup)+"   "+ str((basistotal)/(ftotal)*100.)+" % total \n")
    #timerOut.write("v calc: " +str(vCalcTime)+"   "+str((vcalctotal)/(ftotal)*100.)+" % total \n")
    #timerOut.write("v mat calc: " +str(vtime)+"   "+str((vtotal)/(ftotal)*100.)+" % total \n")
    timerOut.write("v sort: " +str(sorttime)+"   "+str((sorttotal)/(ftotal)*100.)+" % total \n")
    timerOut.write("TR add: " +str(TRtime)+"   "+str((TRtotal)/(ftotal)*100.)+" % total \n")
    timerOut.write("q add: " +str(qtime)+"   "+str((qtotal)/(ftotal)*100.)+" % total \n")
    timerOut.write("R add: " +str(Rtime)+"   "+str((Rtotal)/(ftotal)*100.)+" % total \n")
    timerOut.write("H diag: " +str(Htime)+"   "+str((Htotal)/(ftotal)*100.)+" % total \n")
    #timerOut.write("Sub-total: "+str( (basistotal + vcalctotal + vtotal + sorttotal + TRtotal + qtotal + Rtotal + Htotal)/(ftotal) * 100) + "\n")
    
    timerOut.close()
    #exit()
    # print out the eigenvalues to a textfile
    AuEout = open(labelA+labelB+"_E_Au.txt",'w')
    for s in range(len(AuEVal)):
        AuEout.write(str(AuEVal[s])+" "+str(AuEVal[s]-AgEVal[0]) + "\n")
    AuEout.close()
    
    AgEout = open(labelA+labelB+"_E_Ag.txt",'w')
    for s in range(len(AgEVal)):
        AgEout.write(str(AgEVal[s])+" "+str(AgEVal[s]-AgEVal[0]) + "\n")
    AgEout.close()
    
    BuEout = open(labelA+labelB+"_E_Bu.txt",'w')
    for s in range(len(BuEVal)):
        BuEout.write(str(BuEVal[s])+" "+str(BuEVal[s]-AgEVal[0]) + "\n")
    BuEout.close()
    
    BgEout = open(labelA+labelB+"_E_Bg.txt",'w')
    for s in range(len(BgEVal)):
        BgEout.write(str(BgEVal[s])+" "+str(BgEVal[s]-AgEVal[0]) + "\n")
    BgEout.close()
    
    
    
    # print out the eigenvectors to a textfile
    AuWFout = open(labelAii+labelB+"_wf_Au.txt",'w')
    for s in range(len(AuEVal)):
        for y in range(len(AuEVal)):
            AuWFout.write(str(np.real(AuEVec[y,s]))+" "+str(np.imag(AuEVec[y,s]))+" ")
        AuWFout.write("\n")
    AuWFout.close()
    
    AgWFout = open(labelAii+labelB+"_wf_Ag.txt",'w')
    for s in range(len(AgEVal)):
        for y in range(len(AgEVal)):
            AgWFout.write(str(np.real(AgEVec[y,s]))+" "+str(np.imag(AgEVec[y,s]))+" ")
        AgWFout.write("\n")
    AgWFout.close()
    
    BuWFout = open(labelAii+labelB+"_wf_Bu.txt",'w')
    for s in range(len(BuEVal)):
        for y in range(len(BuEVal)):
            BuWFout.write(str(np.real(BuEVec[y,s]))+" "+str(np.imag(BuEVec[y,s]))+" ")
        BuWFout.write("\n")
    BuWFout.close()
    
    BgWFout = open(labelAii+labelB+"_wf_Bg.txt",'w')
    for s in range(len(BgEVal)):
        for y in range(len(BgEVal)):
            BgWFout.write(str(np.real(BgEVec[y,s]))+" "+str(np.imag(BgEVec[y,s]))+" ")
        BgWFout.write("\n")
    BgWFout.close()
