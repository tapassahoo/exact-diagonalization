import time
from subprocess import call
from os import system
import os
import decimal
 
def jobstring(NameOfServer, src_code, Rpt, jmax, size_grid, dir_output):
	logfile = dir_output+"/diag-submit-p-H2O-Rpt"+str(Rpt)+"ang-j"+str(jmax)+"-grid"+str(size_grid)+".txt"
	jobname = "diag-Rpt"+str(Rpt)

	command_execution = "python " + src_code +"  " +  str(Rpt) + "  " + str(jmax) + "  " + str(size_grid)

	if NameOfServer == "graham":
		account = "#SBATCH --account=rrg-pnroy"
	else:
		account = ""

	job_string = """#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --time=1-00:00
%s
#SBATCH --mem-per-cpu=256GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
%s
""" % (jobname,logfile,account,command_execution)
	return job_string

#initial parameters for qmc.input
size_grid = 4
jrot = 3
status = 'S'
zmin = 2.4
zmax = 10.0
dz = 0.1
nz = int(((zmax-zmin)+dz*0.5)/dz)
nz=nz+1

NameOfServer = 'nlogn'
if status == 'S':
	src_dir = os.getcwd()
	src_code = "/diag_dimer_asym_rot_basis_saved.py"
	dir_output = "/home/tapas/CodesForEigenValues/nonlinear-rotors/exact-energies-of-H2O"
	run_command = src_dir + src_code
	print(run_command)

# Loop over your jobs
for i in range(nz): 

	value = zmin+i*dz
	Rpt = '{:2.1f}'.format(value)
	print(Rpt)

	if status == 'S':
		os.chdir(dir_output)
		#job submission
		fname = 'diag_Rpt'+str(Rpt)+"Ang"
		fwrite = open(fname, 'w')

		fwrite.write(jobstring(NameOfServer, run_command, Rpt, jrot, size_grid, dir_output))
		fwrite.close()
		call(["sbatch", "-p", "highmem", fname])

		os.chdir(src_dir)

	'''
	if status == "A":
		src_file  = "EigenValuesFor2HF-DipoleMoment"+str(DipoleMoment)+"Debye.txt"
		try:
			output    = src_path +folder_run+ "/"+src_file
			call(["cat", output])
		except:
			pass
	'''

'''
call(["rm","*.txt"])
command_run = "./run"
system(command_run)

'''
