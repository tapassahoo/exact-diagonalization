.SUFFIXES: .o .c .f .cxx
options= -O3 -DLINUX -DBLAS
CC=/usr/bin/g++ #/opt/openmpi/bin/mpic++
cc=/usr/bin/gcc #/opt/openmpi/bin/mpicc
f77=gfortran #/opt/openmpi/bin/mpif90
# *******objects for Hui's Lanczos code*******
# objects= matvec.o ran1.o four1.o prod.o gaulegf.o wangpot.o wanghehe.o\
# plgndr.o inter.o  gauleg.o sturmi.o inverr.o inverm.o isoev.o \
# lancbis.o bisec.o genran.o scalar.o hv.o trivec.o peckeris.o
# *******objects for direct diag code*******
objects= matvec.o ran1.o four1.o prod.o  gaulegf.o \
plgndr.o inter.o  gauleg.o genran.o scalar.o cmdstuff.o 
all: H2H2rigid 
H2H2rigid: H2H2rigid.o $(objects) matvec.o
	$(CC) $(options) -o H2H2rigid H2H2rigid.o $(objects) \
-lblas -llapack -L/usr/local/lib -lgfortran
clean:
	rm *.o
.cxx.o: 
	$(CC) -c  $(options) $<
.f.o:
	$(f77) -c  $<
.c.o:
	$(cc) -c $(options) $<
