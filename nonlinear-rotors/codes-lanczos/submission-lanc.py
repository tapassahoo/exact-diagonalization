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
#SBATCH --time=2-00:00
%s
#SBATCH --mem-per-cpu=8GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
%s
""" % (jobname,logfile,account,command_execution)
	return job_string

#initial parameters for qmc.input
status = 'A'
niter = 400
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
	eigval=np.zeros((nz,3))
	err=np.zeros((nz,3))
	saved_Rpt=np.zeros(nz)
	SvNvsRpt=np.zeros(nz)
	S2vsRpt=np.zeros(nz)

# Loop over your jobs
index_end=0
for i in range(nz): 

	value = zmin+i*dz
	Rpt = '{:2.1f}'.format(value)
	print(Rpt)

	if (status == 'S'):
		os.chdir(dir_output)
		#job submission
		fname = 'lanc_Rpt'+str(Rpt)+"Ang-J"+str(jrot)+"-niter"+str(niter)
		if (os.path.isfile(fname) == True):
			print("job name - "+fname)
			print("It has already been submitted")
			os.chdir(src_dir)
			exit()
		else:
			fwrite = open(fname, 'w')

			fwrite.write(jobstring(NameOfServer,Rpt,jrot,dir_output,niter))
			fwrite.close()
			#call(["sbatch", "-p", "highmem", fname])
			call(["sbatch", fname])
			#call(["sbatch", "-C", "faster", fname])

		os.chdir(src_dir)

	if (status == "A"):
		saved_Rpt[i] = Rpt
		if (jrot <= 6):
			thetaNum = int(2*jrot+3)
			angleNum = int(2*(2*jrot+1))
		else:
			thetaNum = int(jrot+2)
			angleNum = int(2*jrot+2)
			
		strFile = "lanc-2-p-H2O-jmax"+str(jrot)+"-Rpt"+str(Rpt)+"Angstrom-grid-"+str(thetaNum)+"-"+str(angleNum)+"-niter"+str(niter)+".txt"

		#fileAnalyze_energy = "ground-state-energy-"+strFile
		fileAnalyze_energy = "energy-levels-"+strFile
		data_input_energy = dir_output+"/"+fileAnalyze_energy
		CMRECIP2KL = 1.4387672

		fileAnalyze_entropy = "ground-state-entropies-"+strFile
		data_input_entropy = dir_output+"/"+fileAnalyze_entropy

		if (os.path.isfile(data_input_energy) == True):
			print(index_end)
			print(data_input_energy)
			eig_kelvin, err_kelvin = np.loadtxt(data_input_energy, usecols=(0, 1), unpack=True)
			for j in range(np.size(eigval,1)):
				eigval[i,j] = eig_kelvin[j]
				err[i,j] = err_kelvin[j]
				print(eigval[i,j])
			#eigvalvsRpt2[i] = eig_kelvin/CMRECIP2KL
		else:
			pass

		index_end = index_end+1

		if (os.path.isfile(data_input_entropy) == True):
			ent = np.loadtxt(data_input_entropy, usecols=(0), unpack=True)
			SvNvsRpt[i] = ent[0]
			S2vsRpt[i] = ent[1]
		else:
			pass


if (status == "A"):
	#printing block is opened
	strFile1 = "lanc-2-p-H2O-jmax"+str(jrot)+"-grid-"+str(thetaNum)+"-"+str(angleNum)+"-niter"+str(niter)+".txt"

	energy_comb1=np.concatenate((eigval[:index_end,:], err[:index_end,:]), axis=1)
	energy_comb=np.concatenate((saved_Rpt[:index_end,np.newaxis],energy_comb1), axis=1)
	eig_file = dir_output+"/ground-state-energy-vs-Rpt-"+strFile1
	np.savetxt(eig_file, energy_comb, fmt='%1.5e', delimiter='    ', header='col1-> Rpt (Angstrom); col2->ground state energy in kelvin; col3->first excited state energy in kelvin; col4->second excited state energy in kelvin. The columns 5-7 are the corresponding errors.')

	#entropy_comb = np.array([saved_Rpt[:index_end-1], SvNvsRpt[:index_end-1], S2vsRpt[:index_end-1]])
	entropy_comb = np.array([saved_Rpt[:index_end], SvNvsRpt[:index_end], S2vsRpt[:index_end]])
	ent_file = dir_output+"/ground-state-entropies-vs-Rpt-"+strFile1
	np.savetxt(ent_file, entropy_comb.T, fmt='%20.8f', delimiter=' ', header='First col. --> Rpt (Angstrom); 2nd and 3rd cols are the S_vN and S_2, respectively. ')
	# printing block is closed

