.SUFFIXES: .o .c .f .cxx

options= -O3 -DLINUX -DBLAS
CC=/usr/bin/g++ #/opt/openmpi/bin/mpic++
cc=/usr/bin/gcc #/opt/openmpi/bin/mpicc
f77=gfortran #/opt/openmpi/bin/mpif90

#objects= matvec.o ran1.o four1.o prod.o \
#plgndr.o inter.o  gauleg.o genran.o scalar.o cmdstuff.o 
objects= matvec.o ran1.o four1.o prod.o \
plgndr.o inter.o  gauleg.o cmdstuff.o 
all: run
run: dimer-diag.o  $(objects) matvec.o
	$(CC) $(options) -o run dimer-diag.o $(objects) \
-lblas -llapack -L/usr/local/lib -lgfortran
clean:
	rm *.o run
.cxx.o: 
	$(CC) -c  $(options) $<
.c.o:
	$(cc) -c $(options) $<
