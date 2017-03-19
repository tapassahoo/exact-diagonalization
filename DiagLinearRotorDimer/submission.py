#!/usr/bin/python
# Example PBS cluster job submission in Python
 
import time
from subprocess import call
from os import system
import os
import decimal
 

def dropzeros(number):
    mynum          = decimal.Decimal(number).normalize()
    # e.g 22000 --> Decimal('2.2E+4')
    return mynum.__trunc__() if not mynum % 1 else float(mynum)


def replace(string_old, string_new, file1, file2):
	'''
	This function replaces old string by new string
	'''
	f1             = open(file1, 'r')
	f2             = open(file2, 'w')
	for line in f1:
		f2.write(line.replace(string_old, string_new))
	f1.close()
	f2.close()


def jobstring(file_name, value, Rpt, DipoleMoment, jrot):
	'''
	This function creats jobstring for #PBS script
	'''
	job_name       = "job_"+str(file_name)+"%d" % value
	walltime       = "400:00:00"
	processors     = "nodes=1:ppn=1"
	command_pimc_run = "./run "+str(Rpt)+" "+str(DipoleMoment)+" "+str(jrot)

	job_string     = """#!/bin/bash
#PBS -N %s
#PBS -l walltime=%s
#PBS -l %s
#PBS -o %s.out
#PBS -e %s.err
export OMP_NUM_THREADS=1
cd $PBS_O_WORKDIR
%s""" % (job_name, walltime, processors, job_name, job_name, command_pimc_run)
	print job_string
	return job_string

#initial parameters for qmc.input
src_path         = "/home/tapas/H2H2/src3/"
nrange           = 20                                                 #param1
Rpt              = 10.05
DipoleMoment     = 0.45
dRpt             = 0.5
jrot             = 3

file1_name       = "result-Rpt"               #param4
argument		 = "Rpt"                                            #param5
status           = "submission"
status           = "analysis"

# Loop over your jobs
for i in range(nrange): 

	#jrot         = "%d" % i
	value        = i*dRpt + 0.5
	Rpt          = '{:2.1f}'.format(value)

	fldr         = file1_name+str(value)
	folder_run   = fldr

	if status == "submission":
		call(["rm", "-rf", folder_run])
		call(["mkdir", folder_run])

		# copy files to running folder
		dest_path    = src_path +folder_run
		source_file  = src_path + "run"
		call(["cp", source_file, dest_path])
	
		os.chdir(dest_path)
		#job submission
		fname        = 'submit_'+str(i)
		fwrite       = open(fname, 'w')

		fwrite.write(jobstring(argument, i, Rpt, DipoleMoment, jrot))
		fwrite.close()
		call(["qsub", fname])

		os.chdir(src_path)

	if status == "analysis":
		output    = src_path +folder_run+"/EigenValuesFor2HF-DipoleMoment0.45.txt"
		call(["cat", output])
'''

call(["rm","*.txt"])
command_run = "./run"
system(command_run)

'''
