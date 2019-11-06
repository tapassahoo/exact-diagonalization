.SUFFIXES: .o .c .f .cxx

options= -O3 -DLINUX -DBLAS
CC=/usr/bin/g++ #/opt/openmpi/bin/mpic++
cc=/usr/bin/gcc #/opt/openmpi/bin/mpicc
f77=gfortran #/opt/openmpi/bin/mpif90

#objects= matvec.o ran1.o four1.o prod.o \
#plgndr.o inter.o  gauleg.o genran.o scalar.o cmdstuff.o 
objects=ran1.o four1.o prod.o gaulegf.o \
plgndr.o centroidstuff.o inter.o  gauleg.o sturmi.o inverr.o inverm.o isoev.o \
lancbis.o bisec.o genran.o scalar.o peckeris.o trivec.o caleng_tip4p_gg.o rotden.o rotpro_sub.o
all: run
run: dimer-lanc.o  $(objects) matvec.o BF.h
	$(CC) $(options) -o run dimer-lanc.o matvec.o $(objects) \
-lblas -llapack -L/usr/local/lib -lgfortran
clean:
	rm *.o run
.cxx.o: 
	$(CC) -c  $(options) $<
.c.o:
	$(cc) -c $(options) $<
