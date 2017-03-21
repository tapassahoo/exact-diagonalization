#!/usr/bin/python

import time
from subprocess import call
from os import system
import os
import decimal
import numpy as np
from numpy import *

nrange       = 20
dRpt         = 0.5
value_min    = 0.5
DipoleMoment = 1.86

file1        = "ev"
file2        = "H"
file3        = "V"
file4        = "log" 
file5        = "EigenValuesFor1HF-DipoleMoment"+str(DipoleMoment)+".txt"
call(['rm', file1, file2, file3, file4, file5])
for i in range(nrange):                                                  #change param13

	value        = i*dRpt + value_min
	Rpt          = '{:2.1f}'.format(value)
	command_linden_run = "./run "+str(Rpt)+"  "+str(DipoleMoment)
	print command_linden_run
	system(command_linden_run)
