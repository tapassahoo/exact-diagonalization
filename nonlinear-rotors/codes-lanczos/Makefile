.SUFFIXES: .o .c .f .cxx


CXX     = g++
CC      = gcc
F77      = gfortran

CFLAGS  = -O3 -DLINUX -DBLAS -fopenmp -lstdc++ #-I /additional/include/dir
CXXFLAGS = -O3 -DLINUX -DBLAS -fopenmp -lstdc++ 
LDFLAGS= -lcblas -llapack -lm -lgfortran 

objects=ran1.o four1.o prod.o gaulegf.o \
plgndr.o inter.o  gauleg.o sturmi.o inverr.o inverm.o isoev.o \
lancbis.o bisec.o genran.o scalar.o trivec.o caleng_qtip4p_rigid.o rotden.o rotpro_sub.o

all: run
run: code-real-basis-lanc-dimer.o  $(objects) matvec.o BF.h
	$(CXX) $(CXXFLAGS) -o run code-real-basis-lanc-dimer.o matvec.o $(objects) $(LDFLAGS)

.cxx.o: 
	$(CXX) -c  $(CXXFLAGS)  $<
.c.o:
	$(CC) -c $(CFLAGS) $<
.f.o:
	$(F77) -c $<

.PHONY: clean
clean:
	rm -f *.o run
