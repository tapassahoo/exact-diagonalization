import time
from subprocess import call
from os import system
import os
import decimal
import numpy as np
 
def jobstring(NameOfServer, src_code, Rpt, jmax, spin_isomer, dir_output):
	logfile = dir_output+"/diag-submit-p-H2O-Rpt"+str(Rpt)+"ang-j"+str(jmax)+".txt"
	jobname = "dg-R"+str(Rpt)

	command_execution = "python " + src_code +"  " +  str(Rpt) + "  " + str(jmax) + " " + spin_isomer

	if NameOfServer == "graham":
		account = "#SBATCH --account=rrg-pnroy"
	else:
		account = ""

	job_string = """#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --time=1-00:00
%s
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
%s
""" % (jobname,logfile,account,command_execution)
	return job_string

#initial parameters for qmc.input
spin_isomer = "para"
status = 'S'
jrot = 12

# making grid points for the intermolecular distance, r
zmin = 2.5
zmax = 2.7
dz = 0.02
nz = int(((zmax-zmin)+dz*0.5)/dz)
nz += 1
zList = [zmin+dz*i for i in range(nz)]
zList += [2.75]
zmin = 2.8
zmax = 5.0
dz = 0.1
nz = int(((zmax-zmin)+dz*0.5)/dz)
nz += 1
zList += [zmin+dz*i for i in range(nz)]
zmin = 5.2
zmax = 10.0
dz = 0.2
nz = int(((zmax-zmin)+dz*0.5)/dz)
nz += 1
print(nz)
zList += [zmin+dz*i for i in range(nz)]
zList = [20.0]

NameOfServer = 'nlogn'
dir_output = "/home/tapas/CodesForEigenValues/nonlinear-rotors/exact-energies-of-H2O"

if (spin_isomer == "spinless"):
	isomer = "-"
if (spin_isomer == "para"):
	isomer = "-p-"
if (spin_isomer == "ortho"):
	isomer = "-o-"

if (status == 'S'):
	src_dir = os.getcwd()
	src_code = "/diag_monomer_asym_rot.py"
	run_command = src_dir + src_code
	print(run_command)

if (status == 'A'):
	eigvalvsRpt1 = np.zeros(len(zList),dtype=float)
	eigvalvsRpt2 = np.zeros(len(zList),dtype=float)
	saved_Rpt = np.zeros(len(zList),dtype=float)
	SvNvsRpt = np.zeros(len(zList),dtype=float)
	S2vsRpt = np.zeros(len(zList),dtype=float)

# Loop over your jobs
index_end=0
for r in zList: 
	Rpt = '{:3.2f}'.format(r)

	if (status == 'S'):
		os.chdir(dir_output)
		#job submission
		fname = 'job-diag-R'+str(Rpt)+"Ang-Jmax"+str(jrot)
		if (os.path.isfile(fname)==True):
			print("job name - "+fname)
			print("It has already been submitted.")
			os.chdir(src_dir)
			exit()
		else:
			fwrite=open(fname, 'w')
			fwrite.write(jobstring(NameOfServer, run_command, Rpt, jrot, spin_isomer, dir_output))
			fwrite.close()
			#call(["sbatch", "-p", "highmem", fname])
			call(["sbatch", "-C", "faster", fname])

		os.chdir(src_dir)

	if (status == "A"):
		saved_Rpt[index_end] = float(Rpt)
		thetaNum = int(2*jrot+3)
		angleNum = int(2*(2*jrot+1))
			
		strFile = "of-2"+isomer+"H2O-one-rotor-fixed-cost1-jmax"+str(jrot)+"-Rpt"+str(Rpt)+"Angstrom-grids-"+str(thetaNum)+"-"+str(angleNum)+"-diag.txt"
		#strFile = "of-1"+isomer+"H2O-jmax"+str(jrot)+"-Rpt"+str(Rpt)+"Angstrom-grids-"+str(thetaNum)+"-"+str(angleNum)+"-diag.txt"

		fileAnalyze_energy = "ground-state-energy-"+strFile
		data_input_energy = dir_output+"/"+fileAnalyze_energy
		CMRECIP2KL = 1.4387672

		fileAnalyze_entropy = "ground-state-entropies-"+strFile
		data_input_entropy = dir_output+"/"+fileAnalyze_entropy

		#print(data_input_energy)
		if (os.path.isfile(data_input_energy) == True):
			eig_kelvin, eig_wavenumber = np.loadtxt(data_input_energy, usecols=(0, 1), unpack=True)
			#print(eig_kelvin, eig_wavenumber)
			eigvalvsRpt1[index_end] = eig_kelvin
			eigvalvsRpt2[index_end] = eig_wavenumber
		else:
			pass

		print(index_end)
		index_end = index_end+1

if (status == "A"):
	#printing block is opened
	strFile1 = "of-2"+isomer+"H2O-one-rotor-fixed-cost1-jmax"+str(jrot)+"-grid-"+str(thetaNum)+"-"+str(angleNum)+"-diag.txt"
	#strFile1 = "of-1"+isomer+"H2O-jmax"+str(jrot)+"-grid-"+str(thetaNum)+"-"+str(angleNum)+"-diag.txt"

	energy_comb = np.array([saved_Rpt, eigvalvsRpt1, eigvalvsRpt2])
	eig_file = dir_output+"/ground-state-energy-vs-Rpt-"+strFile1
	np.savetxt(eig_file, energy_comb.T, fmt='%20.8f', delimiter=' ', header='First col. --> Rpt (Angstrom); 2nd and 3rd cols are the eigen values of (Htot = Hrot + Hvpot) in Kelvin and wavenumber, respectively. ')

	'''
	entropy_comb = np.array([saved_Rpt, SvNvsRpt, S2vsRpt])
	ent_file = dir_output+"/ground-state-entropies-vs-Rpt-"+strFile1
	np.savetxt(ent_file, entropy_comb.T, fmt='%20.8f', delimiter=' ', header='First col. --> Rpt (Angstrom); 2nd and 3rd cols are the S_vN and S_2, respectively. ')
	# printing block is closed
	'''

