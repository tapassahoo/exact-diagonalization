#!/usr/bin/python
 
import time
from subprocess import call
from os import system
import os
import decimal
import numpy as np
from numpy import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from os import system
from sys import argv

nbins       = 50          #change
srcfile     = "/work/tapas/linear_rotors/Rpt7.0Angstrom-DipoleMoment1.86Debye-tau0.002Kinv-Blocks10000-System2HF-e0vsbeads51/results/pigs_instant.dof"
srcfile1     = "/work/tapas/linear_rotors/Rpt10.0Angstrom-DipoleMoment1.86Debye-tau0.002Kinv-Blocks10000-System2HF-e0vsbeads51/results/pigs_instant.dof"
srcfile_exact = "DataForHistogramFor1HF-Rpt10Angstrom-DipoleMoment1.86Debye.txt"
col_costheta = loadtxt(srcfile, unpack=True, usecols=[1])
col_costheta1 = loadtxt(srcfile1, unpack=True, usecols=[1])
hist, bins = np.histogram(col_costheta,nbins,density=True)
print len(hist), len(bins)
print np.sum(hist)
print np.sum(hist*np.diff(bins))

plt.hist(col_costheta, bins=20, histtype='stepfilled', normed=True, color='b', label='Gaussian')
plt.hist(col_costheta1, bins=20, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='Uniform')

#plt.hist( col_costheta, bins=nbins, normed = 1, histtype='stepfilled', color='b')  # plt.hist passes it's arguments to np.histogram
#plt.hist( col_costheta1, bins=nbins, normed = 1, histtype='stepfilled', color='r')
cost, dist = loadtxt(srcfile_exact, unpack=True, usecols=[0, 1])
#plt.plot(cost, dist, linestyle = '-', color = 'black', label = 'Exact', lw = 3)
plt.grid(True)

plt.xlabel(r'$\cos(\theta)$', fontsize = 20)
plt.ylabel('Density', fontsize = 20)
#plt.xlim(-0.01,1.01)
plt.grid(True)
plt.show()
