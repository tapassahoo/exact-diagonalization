import time
from subprocess import call
from os import system
import os
import decimal
import numpy as np
 
def jobstring(NameOfServer,Rpt,jmax,dir_output,niter):
	logfile = dir_output+"/lanc-submit-p-H2O-Rpt"+str(Rpt)+"ang-j"+str(jmax)+"-niter"+str(niter)+".txt"
	jobname = "lcR"+str(Rpt)

	command_execution = "./run  " +  str(Rpt) + "  " + str(jmax) + "  " + str(niter)

	if NameOfServer == "graham":
		account = "#SBATCH --account=rrg-pnroy"
	else:
		account = ""

	job_string = """#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --time=1-00:00
%s
#SBATCH --mem-per-cpu=8GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
%s
""" % (jobname,logfile,account,command_execution)
	return job_string

#initial parameters for qmc.input
status = 'A'
niter = 100
jrot = 2
zmin = 2.6
zmax = 10.0
dz = 0.1
nz = int(((zmax-zmin)+dz*0.5)/dz)
nz=nz+1

NameOfServer = 'nlogn'
dir_output = "/home/tapas/CodesForEigenValues/nonlinear-rotors/exact-energies-of-H2O"

if (status == 'S'):
	src_dir = os.getcwd()
	src_code = "/run"
	run_command = src_dir + src_code
	call(["cp", run_command, dir_output])
	print(run_command)

if (status == 'A'):
	eigvalvsRpt1 = np.zeros(nz)
	eigvalvsRpt2 = np.zeros(nz)
	saved_Rpt = np.zeros(nz)
	SvNvsRpt = np.zeros(nz)
	S2vsRpt = np.zeros(nz)

# Loop over your jobs
index_end=0
for i in range(nz): 

	value = zmin+i*dz
	Rpt = '{:2.1f}'.format(value)
	print(Rpt)

	if (status == 'S'):
		os.chdir(dir_output)
		#job submission
		fname = 'lanc_Rpt'+str(Rpt)+"Ang"
		fwrite = open(fname, 'w')

		fwrite.write(jobstring(NameOfServer,Rpt,jrot,dir_output,niter))
		fwrite.close()
		#call(["sbatch", "-p", "highmem", fname])
		call(["sbatch", fname])

		os.chdir(src_dir)

	if (status == "A"):
		saved_Rpt[i] = Rpt
		thetaNum = int(2*jrot+3)
		angleNum = int(2*(2*jrot+1))
		strFile = "lanc-2-p-H2O-jmax"+str(jrot)+"-Rpt"+str(Rpt)+"Angstrom-grid-"+str(thetaNum)+"-"+str(angleNum)+"-niter"+str(niter)+".txt"

		fileAnalyze_energy = "ground-state-energy-"+strFile
		data_input_energy = dir_output+"/"+fileAnalyze_energy
		CMRECIP2KL = 1.4387672
		if (os.path.isfile(data_input_energy) == True):
			eig_kelvin= np.loadtxt(data_input_energy, usecols=(0), unpack=True)
			eigvalvsRpt1[i] = eig_kelvin
			eigvalvsRpt2[i] = eig_kelvin/CMRECIP2KL
			index_end = index_end+1
			print(index_end)
		else:
			break

		'''
		fileAnalyze_entropy = "ground-state-entropies-"+strFile
		data_input_entropy = dir_output+"/"+fileAnalyze_entropy
		ent = np.loadtxt(data_input_entropy, usecols=(0), unpack=True)
		SvNvsRpt[i] = ent[0]
		S2vsRpt[i] = ent[1]
		'''

if (status == "A"):
	#printing block is opened
	strFile1 = "lanc-2-p-H2O-jmax"+str(jrot)+"-grid-"+str(thetaNum)+"-"+str(angleNum)+"-niter"+str(niter)+".txt"

	energy_comb = np.array([saved_Rpt[:index_end], eigvalvsRpt1[:index_end], eigvalvsRpt2[:index_end]])
	eig_file = dir_output+"/ground-state-energy-vs-Rpt-"+strFile1
	np.savetxt(eig_file, energy_comb.T, fmt='%20.8f', delimiter=' ', header='First col. --> Rpt (Angstrom); 2nd and 3rd cols are the eigen values of (Htot = Hrot + Hvpot) in Kelvin and wavenumber, respectively. ')

	#entropy_comb = np.array([saved_Rpt, SvNvsRpt, S2vsRpt])
	#ent_file = dir_output+"/ground-state-entropies-vs-Rpt-"+strFile1
	#np.savetxt(ent_file, entropy_comb.T, fmt='%20.8f', delimiter=' ', header='First col. --> Rpt (Angstrom); 2nd and 3rd cols are the S_vN and S_2, respectively. ')
	# printing block is closed

