import time
from subprocess import call
from os import system
import os
import decimal
 
def jobstring(NameOfServer, Rpt, jmax, niter):

	logfile = "log-p-H2O-Rpt"+str(Rpt)+"ang-j"+str(jmax)+"-niter"+str(niter)+".txt"
	jobname = "Rpt"+str(Rpt)

	command_execution = "./run  " +  str(Rpt) + "  " + str(jmax) + "  " + str(niter)

	if NameOfServer == "graham":
		account = "#SBATCH --account=rrg-pnroy"
	else:
		account = ""

	job_string = """#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --time=7-00:00
%s
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
%s
""" % (jobname,logfile,account,command_execution)
	return job_string

#initial parameters for qmc.input
NameOfServer = 'nlogn'
niter = 0
src_dir = os.getcwd()
jrot = 2

status = 'S'
zmin = 2.5
zmax = 10.0
dz = 0.1
nz = int(((zmax-zmin)+dz*0.5)/dz)
nz=nz+1

dir_output = "Results-p-H2O/"

if status == 'S':
	#call(["rm", "-rf", dir_output])
	#call(["mkdir", dir_output])
	dest_path = src_dir + "/" + dir_output
	source_file = src_dir + "/run"
	call(["cp", source_file, dest_path])

# Loop over your jobs
for i in range(nz): 

	value = zmin+i*dz
	Rpt = '{:2.1f}'.format(value)
	print(Rpt)

	if status == 'S':
		os.chdir(dest_path)
		#job submission
		fname = 'submit_Rpt'+str(Rpt)+"Ang"
		fwrite = open(fname, 'w')

		fwrite.write(jobstring(NameOfServer, Rpt, jrot, niter))
		fwrite.close()
		call(["sbatch", fname])

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
