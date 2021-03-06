import time
from subprocess import call
from os import system
import os
import decimal
import numpy as np
 
def jobstring(NameOfServer, src_code, Rpt, jmax, size_grid, dir_output):
	logfile = dir_output+"/arpack-submit-p-H2O-Rpt"+str(Rpt)+"ang-j"+str(jmax)+"-grid"+str(size_grid)+"-qTIP4P.txt"
	#logfile = dir_output+"/arpack-submit-p-H2O-Rpt"+str(Rpt)+"ang-j"+str(jmax)+"-grid"+str(size_grid)+".txt"
	jobname = "qTIP4P"+str(Rpt)

	command_execution = "time python " + src_code +"  " +  str(Rpt) + "  " + str(jmax)

	if NameOfServer == "graham":
		account = "#SBATCH --account=rrg-pnroy"
	else:
		account = ""

	job_string = """#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --time=7-00:00
%s
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
%s
""" % (jobname,logfile,account,command_execution)
	return job_string

#initial parameters for qmc.input
status = 'A'
grid_increment = 0
jrot = 4
zmin = 2.6
zmax = 10.0
dz = 0.1
nz = int(((zmax-zmin)+dz*0.5)/dz)
nz=nz+1

NameOfServer = 'nlogn'
dir_output = "/home/tapas/CodesForEigenValues/nonlinear-rotors/exact-energies-of-H2O"

if (status == 'S'):
	src_dir = os.getcwd()
	src_code = "/lanczos-arpack-scipy-original.py"
	run_command = src_dir + src_code
	print(run_command)

if (status == 'A'):
	eigvalvsRpt1 = np.zeros(nz)
	eigvalvsRpt2 = np.zeros(nz)
	saved_Rpt = np.zeros(nz)
	#SvNvsRpt = np.zeros(nz)
	#S2vsRpt = np.zeros(nz)

# Loop over your jobs
for i in range(nz): 

	value = zmin+i*dz
	Rpt = '{:2.1f}'.format(value)
	print(Rpt)

	if (status == 'S'):
		os.chdir(dir_output)
		#job submission
		fname = 'qTIP4P-Rpt'+str(Rpt)+"Ang"
		#fname = 'Rpt'+str(Rpt)+"Ang"
		fwrite = open(fname, 'w')

		fwrite.write(jobstring(NameOfServer, run_command, Rpt, jrot, grid_increment, dir_output))
		fwrite.close()
		call(["sbatch", "-C", "faster", fname])

		os.chdir(src_dir)

	if (status == "A"):
		saved_Rpt[i] = Rpt
		thetaNum = int(2*jrot+1+grid_increment)
		angleNum = int(2*(2*jrot+1)+grid_increment)
		#angleNum = int((2*jrot+1)+grid_increment)
		strFile = "arpack-2-p-H2O-jmax"+str(jrot)+"-Rpt"+str(Rpt)+"Angstrom-grid"+str(thetaNum)+"-"+str(angleNum)+"-qTIP4P.txt"

		fileAnalyze_energy = "energy-levels-"+strFile
		data_input_energy = dir_output+"/"+fileAnalyze_energy
		eig_kelvin = np.loadtxt(data_input_energy)

		eigvalvsRpt1[i] = eig_kelvin
		eigvalvsRpt2[i] = 0.0
		'''
		eigvalvsRpt1[i] = eig_kelvin[0]
		eigvalvsRpt2[i] = eig_kelvin[1]
		'''

		'''
		fileAnalyze_entropy = "ground-state-entropies-"+strFile
		data_input_entropy = dir_output+"/"+fileAnalyze_entropy
		ent = np.loadtxt(data_input_entropy, usecols=(0), unpack=True)
		SvNvsRpt[i] = ent[0]
		S2vsRpt[i] = ent[1]
		'''

if (status == "A"):
	#printing block is opened
	strFile1 = "arpack-2-p-H2O-jmax"+str(jrot)+"-grid-"+str(thetaNum)+"-"+str(angleNum)+"-qTIP4P.txt"

	energy_comb = np.array([saved_Rpt, eigvalvsRpt1, eigvalvsRpt2])
	eig_file = dir_output+"/ground-state-energy-vs-Rpt-"+strFile1
	np.savetxt(eig_file, energy_comb.T, fmt='%20.8f', delimiter=' ', header='First col. --> Rpt (Angstrom); 2nd and 3rd cols are ground state and first excited energies in Kelvin, respectively. ')

	'''
	entropy_comb = np.array([saved_Rpt, SvNvsRpt, S2vsRpt])
	ent_file = dir_output+"/ground-state-entropies-vs-Rpt-"+strFile1
	np.savetxt(ent_file, entropy_comb.T, fmt='%20.8f', delimiter=' ', header='First col. --> Rpt (Angstrom); 2nd and 3rd cols are the S_vN and S_2, respectively. ')
	# printing block is closed

	'''
