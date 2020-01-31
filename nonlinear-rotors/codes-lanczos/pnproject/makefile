.SUFFIXES: .o .c .f .cxx
# -OPT:fast_sqrt=on:IEEE_arith=3:roundoff=3
#LFLAGS = -L/v/pnroy/interface/qt/lib -lqt
INCDIR = -I/home/pnroy/exact_trimer/mbpol2
options=-DLINUX -DBLAS -O3 $(INCDIR)
CC=g++
cc=gcc
f77=gfortran
objects=ran1.o four1.o prod.o gaulegf.o \
plgndr.o centroidstuff.o inter.o  gauleg.o sturmi.o inverr.o inverm.o isoev.o \
lancbis.o bisec.o genran.o scalar.o peckeris.o trivec.o
grid: grid_for_tom.o $(objects) matvec.o BF.h
	$(CC)  -o grid grid_for_tom.o matvec.o $(objects) \
-lgfortran -llapack -lblas
trimer: trimer.o $(objects) matvec.o BF.h
	$(CC)  -o trimer trimer.o matvec.o $(objects) \
-lgfortran -llapack -lblas -L/home/pnroy/exact_trimer/mbpol2 -lmbpol
clean:
	rm *.o
Helium.o : Helium.h Helium.cxx
	$(CC) -c $(options) Helium.cxx
mark.o : BF.h mark.cxx
	$(CC) -c $(options) mark.cxx
.cxx.o: 
	$(CC)  -c  $(options) $<
.f.o:
	$(f77) -c  $<
.c.o:
	$(cc) -c $(options) $<
