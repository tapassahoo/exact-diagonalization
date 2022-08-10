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
	logfile=dir_output+"/lanc-submit-p-H2O-Rpt"+str(Rpt)+"ang-j"+str(jmax)+"-niter"+str(niter)+".txt"
	jobname="lcR"+str(Rpt)

	command_execution="time ./run  "+str(Rpt)+"  "+str(jmax)+"  "+str(niter)+"  "+str(emin)+"  "+str(emax)

        ystem_name = os.getenv('HOSTNAME')
	if (NameOfServer=="graham"):
		account="#SBATCH --account=rrg-pnroy"
	else:
		account=""

	job_string="""#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --time=14-00:00
%s
#SBATCH --mem-per-cpu=6GB
#SBATCH --cpus-per-task=16
export OMP_NUM_THREADS=16
%s
""" % (jobname,logfile,account,command_execution)
	return job_string


#initial parameters for qmc.input
status = 'S'
niter = 10
jrot = 10

# making grid points for the intermolecular distance, r
'''
zmin = 2.5
zmax = 2.7
dz = 0.02
nz = int(((zmax-zmin)+dz*0.5)/dz)
nz += 1
zList = [zmin+dz*i for i in range(nz)]
zList += [2.75]
zmin = 2.8
zmax = 4.0
dz = 0.1
nz = int(((zmax-zmin)+dz*0.5)/dz)
nz += 1
zList += [zmin+dz*i for i in range(nz)]
zmin = 5.2
zmax = 10.0
dz = 0.2
nz = int(((zmax-zmin)+dz*0.5)/dz)
nz += 1
zList += [zmin+dz*i for i in range(nz)]
print(nz)
'''
zmin = 10.0
zmax = 10.0
dz = 1.0
nz = int(((zmax-zmin)+dz*0.5)/dz)
nz += 1
zList = [zmin+dz*i for i in range(nz)]
print(nz)

NameOfServer='nlogn'
dir_output="/home/tapas/CodesForEigenValues/nonlinear-rotors/exact-energies-of-H2O"

if (status=='S'):
	src_dir=os.getcwd()
	src_code="/run"
	run_command=src_dir + src_code
	call(["cp", run_command, dir_output])
	print(run_command)
	#read the pigsdata.txt
	rpt_pigs,eng_pigs,err_pigs=np.loadtxt('pigsdata.txt', usecols=(0,1,2), unpack=True)

if (status=='A'):
	numb_states=1
	eigval=np.zeros(len(zList),dtype=float)
	err=np.zeros(len(zList),dtype=float)
	saved_Rpt=np.zeros(len(zList),dtype=float)
	#SvNvsRpt=np.zeros(len(zList),dtype=float)
	#S2vsRpt=np.zeros(len(zList),dtype=float)

# Loop over your jobs
index_end=0
for r in zList: 

	Rpt="{:3.2f}".format(r)

	if (status=='S'):
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
			get_index=np.where(rpt_pigs==float(Rpt))[0][0]
			emin=eng_pigs[get_index]-3.0*err_pigs[get_index]
			emax=0.0#eng_pigs[get_index]+3.0*err_pigs[get_index]
			fwrite=open(fname, 'w')
			fwrite.write(jobstring(NameOfServer,Rpt,jrot,dir_output,niter,emin,emax))
			fwrite.close()
			#call(["sbatch", "-p", "highmem", fname])
			#call(["sbatch", fname])
			call(["sbatch", "-C", "faster", fname])

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
