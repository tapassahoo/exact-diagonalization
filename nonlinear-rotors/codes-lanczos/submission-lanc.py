import time
from subprocess import call
from os import system
import os
import decimal
import numpy as np

def dropzeros(number):
	mynum          = decimal.Decimal(number).normalize()
	# e.g 22000 --> Decimal('2.2E+4')
	return mynum.__trunc__() if not mynum % 1 else float(mynum) 

def jobstring(NameOfServer,Rpt,jmax,dir_output,niter,emin,emax):
	'''
	The function generates a script file based on SLURM scheduler for running the jobs on computecanada server.
	'''
	logfile=dir_output+"/lanc-submit-p-H2O-Rpt"+str(Rpt)+"ang-j"+str(jmax)+"-niter"+str(niter)+".txt"
	jobname="lcR"+str(Rpt)

	command_execution="time ./run  "+str(Rpt)+"  "+str(jmax)+"  "+str(niter)+"  "+str(emin)+"  "+str(emax)

	if (NameOfServer=="computecanada"):
		account="#SBATCH --account=rrg-pnroy"
	else:
		account=""

	job_string="""#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --time=00-00:30
%s
#SBATCH --mem-per-cpu=2GB
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=1
%s
""" % (jobname,logfile,account,command_execution)
	return job_string

# Critical parameters to execute the script
status = "S"
niter = 100
jrot = 2

# Grid points for the intermolecular distance are
zList1 = np.arange(2.5, 2.7001, 0.02)
zList1 = np.append(zList1, [2.75])
zList2 = np.arange(2.8, 4.01, 0.1)
zList2 = np.append(zList1, zList2)
zList3 = np.arange(4.2, 10.01, 0.2)
zList = np.append(zList2, zList3)
numb_jobs = zList.size
print("The number of jobs that will be running is ", numb_jobs)

# Determination of the output directory based on server
server_name = "computecanada"#os.getenv('HOSTNAME').split('.')[2]
if (server_name == "computecanada"):
	dir_output="/scratch/tapas/exact-results/"
else:
	home=os.path.expanduser("~")
	dir_output=home+"outputs/exact-results/"

if not os.path.isdir(dir_output):
	os.mkdir(dir_output)


if (os.path.exists(dir_output) == True):
	str_out_dir = "The output files are stored in "+dir_output
else:
	str_out_dir = "First make "+dir_output+" directory."

print(str_out_dir)

if (status == "S"):
	src_dir=os.getcwd()
	src_code="/run"
	run_command=src_dir + src_code
	call(["cp", run_command, dir_output])
	#read the pigsdata.txt
	#rpt_pigs,eng_pigs,err_pigs=np.loadtxt('pigsdata.txt', usecols=(0,1,2), unpack=True)

if (status=="A"):
	numb_states=1
	eigval=np.zeros(len(zList),dtype=float)
	err=np.zeros(len(zList),dtype=float)
	saved_Rpt=np.zeros(len(zList),dtype=float)
	#SvNvsRpt=np.zeros(len(zList),dtype=float)
	#S2vsRpt=np.zeros(len(zList),dtype=float)

index_end=0
for r in zList: 

	Rpt="{:3.2f}".format(r)

	if (status == "S"):
		os.chdir(dir_output)

		#job submission
		fname = 'job-lanc-R'+str(Rpt)+"Ang-Jmax"+str(jrot)+"-niter"+str(niter)
		if (os.path.isfile(fname)==True):
			print("job name - "+fname)
			print("It has already been submitted.")
			os.chdir(src_dir)
			exit()
		else:
			#find the emin and emax vaule at the Rpt 
			#get_index=np.where(rpt_pigs==float(Rpt))[0][0]
			emin=-10.0#eng_pigs[get_index]-3.0*err_pigs[get_index]
			emax=0.0#eng_pigs[get_index]+3.0*err_pigs[get_index]
			fwrite=open(fname, 'w')
			fwrite.write(jobstring(server_name,Rpt,jrot,dir_output,niter,emin,emax))
			fwrite.close()
			#call(["sbatch", "-p", "highmem", fname])
			call(["sbatch", fname])
			#call(["sbatch", "-C", "faster", fname])

		os.chdir(src_dir)

	if (status == "A"):
		saved_Rpt[index_end] = float(Rpt)
		if (jrot <= 4):
			thetaNum = int(2*jrot+2)
			angleNum = int(2*(2*jrot+2))
		else:
			thetaNum = 20
			angleNum = 20 #int(2*(2*jrot+2))
			
		strFile = "lanc-2-p-H2O-jmax"+str(jrot)+"-Rpt"+str(Rpt)+"Angstrom-grid-"+str(thetaNum)+"-"+str(angleNum)+"-niter"+str(niter)+".txt"

		fileAnalyze_energy = "ground-state-energy-"+strFile
		data_input_energy = dir_output+"/"+fileAnalyze_energy
		CMRECIP2KL = 1.4387672

		fileAnalyze_entropy = "ground-state-entropies-"+strFile
		data_input_entropy = dir_output+"/"+fileAnalyze_entropy

		if (os.path.isfile(data_input_energy) == True):
			eig_kelvin, err_kelvin = np.loadtxt(data_input_energy, usecols=(0, 1), unpack=True)
			print(eig_kelvin, err_kelvin)
			eigval[index_end]=eig_kelvin
			err[index_end]=err_kelvin
		else:
			pass


		'''
		if (os.path.isfile(data_input_entropy) == True):
			ent = np.loadtxt(data_input_entropy, usecols=(0), unpack=True)
			SvNvsRpt[index_end] = ent[0]
			S2vsRpt[index_end] = ent[1]
		else:
			pass
		'''

		print(index_end)
		index_end = index_end+1

if (status == "A"):
	#printing block is opened
	strFile1 = "lanc-2-p-H2O-jmax"+str(jrot)+"-grid-"+str(thetaNum)+"-"+str(angleNum)+"-niter"+str(niter)+".txt"

	energy_comb = np.array([saved_Rpt, eigval, err])
	eig_file = dir_output+"/ground-state-energy-vs-Rpt-"+strFile1
	np.savetxt(eig_file, energy_comb.T, fmt='%20.8f', delimiter=' ', header='First col. --> Rpt (Angstrom); 2nd and 3rd cols are ground state energy and the corresponding error in Kelvin, respectively. ')
	print(eig_file)

	'''
	#entropy_comb = np.array([saved_Rpt[:index_end-1], SvNvsRpt[:index_end-1], S2vsRpt[:index_end-1]])
	entropy_comb = np.array([saved_Rpt[:index_end], SvNvsRpt[:index_end], S2vsRpt[:index_end]])
	ent_file = dir_output+"/ground-state-entropies-vs-Rpt-"+strFile1
	np.savetxt(ent_file, entropy_comb.T, fmt='%20.8f', delimiter=' ', header='First col. --> Rpt (Angstrom); 2nd and 3rd cols are the S_vN and S_2, respectively. ')
	# printing block is closed
	'''
