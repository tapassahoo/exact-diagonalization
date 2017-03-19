// useful functions below
#include "cmdstuff.h"

// temp: exact HO
double HOrho(double xc,double pc,double beta,double m,double w)
{
  double pwr=beta*.5*(m*w*w*xc*xc+pc*pc/m);
  double rhoexp=exp(-pwr);
  double trg=sinh(w*beta*.5);
  double rhofac=w*beta*.5/trg;
  double rho=rhoexp*rhofac;
  return rho;
}

double HOrho_E(double xc,double pc,double beta,double m,double w)
{
  double alpha=1./tanh(w*beta*.5)-2./(w*beta);
  double pwr1=(1./alpha+beta*w*.5)*m*w*xc*xc;
  double pwr2=(1/(w*alpha)+beta*.5)*pc*pc/m;
  double pwr=pwr1+pwr2;
  double rhoexp=exp(-pwr);
  double trg=sinh(w*beta*.5);
  double rhofac=w*beta*.5/(trg*alpha);
  double rho=rhoexp*rhofac;
  return rho;
}
// end temp

double potentialplot(double x,double a,double b,double c)
{
  return (a*x*x+b*pow(x,3.)+c*pow(x,4.));
}

//buck potential
double Vbuck(double r){
  //buch et al. 1982, r in ang
  double f,V;
  double a=101.4;       //eV
  double beta = 2.779;  //ang-1
  double alpha=0.08;    //ang-2
  double c6=7.254;      //eV*ang6
  double c8=36.008;     //eV*ang8
  double c10=225.56;    //eV*ang10
  double d=5.102;       //ang
  r/=atob;
  f = 1.0;
  if (r <= d) f = exp(-1*pow((d/r-1),2));
  // V in Hartree
  V = (1./27.2113845)*(a*exp((-1.*beta*r)-(alpha*pow(r,2))) - f*((c6/pow(r,6)) + (c8/pow(r,8)) + (c10/pow(r,10))));
  return V;
}

matrix xharmonic(int size)
{
  // dimensionless X for harmonic oscillator
  matrix xexact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      xexact(i,j)=0.;
      if (j == (i+1))
	xexact(i,j)=sqrt((double)i+1.);
      if (j == (i-1))
	xexact(i,j)=sqrt((double)i); 
      xexact(i,j)*=sqrt(.5);
    }
  return xexact;
}

matrix xharmonic_S(int size)
{
  // a block matrix: S(columns) and A(rows)
  matrix xexact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      xexact(i,j)=0.;
      if (j == i)
	xexact(i,j)=sqrt((double)i*2.+1.);
      if (j == (i+1))
	xexact(i,j)=sqrt((double)i*2.+2.);
      xexact(i,j)*=sqrt(.5);
    }
  return xexact;
}

matrix xharmonic_A(int size)
{
  // a block matrix: A(columns) and S(rows)
  matrix xexact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      xexact(i,j)=0.;
      if (j == i)
	xexact(i,j)=sqrt((double)i*2.+1.);
      if (j == (i-1))
	xexact(i,j)=sqrt((double)i*2.);
      xexact(i,j)*=sqrt(.5);
    }
  return xexact;
}

matrix x2harmonic(int size)
{
  matrix x2exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      x2exact(i,j)=0.;
      if (j == i)
        x2exact(i,j)=2.*(double)i+1.;
      if (j == (i+2))
        x2exact(i,j)=sqrt(((double)i+1.)*((double)i+2.));
      if (j == (i-2))
       	x2exact(i,j)=sqrt((double)i*((double)i-1.));
      x2exact(i,j)*=.5;
    }
  return x2exact;
}

matrix x2harmonic_S(int size)
  // for x2exact, change i from x2harmonic to 2i.
  // (j==(i+1)) here corresponds to (j==(i+2)) in x2harmonic, etc.
{
  matrix x2exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      x2exact(i,j)=0.;
      if (j == i)
        x2exact(i,j)=4.*double(i)+1.;
      if (j == (i+1))
	x2exact(i,j)=sqrt(((double)i*2.+1.)*((double)i*2.+2.));
      if (j == (i-1))
	x2exact(i,j)=sqrt(((double)i*2.)*((double)i*2.-1.));
      x2exact(i,j)*=.5;
    }
  return x2exact;
}

matrix x2harmonic_A(int size)
  // for x2exact, change i from x2harmonic to 2i+1.
  // (j==(i+1)) here corresponds to (j==(i+2)) in x2harmonic, etc.
{
  matrix x2exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      x2exact(i,j)=0.;
      if (j == i)
        x2exact(i,j)=4.*double(i)+3.;
      if (j == (i+1))
	x2exact(i,j)=sqrt(((double)i*2.+2.)*((double)i*2.+3.));
      if (j == (i-1))
	x2exact(i,j)=sqrt(((double)i*2.)*((double)i*2.+1.));
      x2exact(i,j)*=.5;
    }
  return x2exact;
}

matrix x3harmonic(int size)
{
  matrix x3exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      x3exact(i,j)=0.;
      if (j == (i+3))
	x3exact(i,j)=sqrt(((double)i+1.)*((double)i+2.)*((double)i+3.));
      if (j == (i+1))
	x3exact(i,j)=(3.*(double)i+3.)*sqrt((double)i+1.);
      if (j == (i-1))
	x3exact(i,j)=3.*((double)i)*sqrt((double)i); 
      if (j == (i-3))
	x3exact(i,j)=sqrt((double)i*((double)i-1.)*((double)i-2.));
      x3exact(i,j)*=.5*sqrt(.5);
    }
  return x3exact;
}

matrix x3harmonic_S(int size)
{
  // a block matrix: S(columns) and A(rows)
  matrix x3exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      x3exact(i,j)=0.;
      if (j == i)
	x3exact(i,j)=3.*((double)i*2.+1.)*sqrt((double)i*2.+1.);
      if (j == (i+2))
	x3exact(i,j)=sqrt(((double)i*2.+2.)*((double)i*2.+3.)*((double)i*2.+4.));
      if (j == (i+1))
	x3exact(i,j)=3.*((double)i*2.+2.)*sqrt((double)i*2.+2.);
      if (j == (i-1))
	x3exact(i,j)=sqrt(((double)i*2.-1.)*((double)i*2.)*((double)i*2.+1.));
      x3exact(i,j)*=.5*sqrt(.5);
    }
  return x3exact;
}

matrix x3harmonic_A(int size)
{
  // a block matrix: A(columns) and S(rows)
  matrix x3exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      x3exact(i,j)=0.;
      if (j == i)
	x3exact(i,j)=3.*((double)i*2.+1.)*sqrt((double)i*2.+1.);
      if (j == (i+1))
	x3exact(i,j)=sqrt(((double)i*2.+1.)*((double)i*2.+2.)*((double)i*2.+3.));
      if (j == (i-1))
	x3exact(i,j)=3.*((double)i*2.)*sqrt((double)i*2.);
      if (j == (i-2))
	x3exact(i,j)=sqrt((double)i*2.*((double)i*2.-1.)*((double)i*2.-2.));
      x3exact(i,j)*=.5*sqrt(.5);
    }
  return x3exact;
}

matrix x4harmonic(int size)
{
  matrix x4exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      x4exact(i,j)=0.;
      if (j == i)
	x4exact(i,j)=6.*((double)(i*i)+(double)i)+3.;
      if (j == (i+4))
	x4exact(i,j)=sqrt(((double)i+1.)*((double)i+2.)*((double)i+3.)*((double)i+4.));
      if (j == (i+2))
	x4exact(i,j)=(4.*(double)i+6.)*sqrt(((double)i+1.)*((double)i+2.));
      if (j == (i-2))
	x4exact(i,j)=(4.*(double)i-2.)*sqrt((double)i*((double)i-1.));
      if (j == (i-4))
	x4exact(i,j)=sqrt((double)i*((double)i-1.)*((double)i-2.)*((double)i-3.));
      x4exact(i,j)*=.25;
    }
  return x4exact;
}

matrix x4harmonic_S(int size)
  // for x4exact, change i from x4harmonic to 2i.
  // (j==(i+2)) here corresponds to (j==(i+4)) in x4harmonic, etc.
{
  matrix x4exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      x4exact(i,j)=0.;
      if (j == i)
	x4exact(i,j)=12.*((double)(2*i*i)+(double)i)+3.;
      if (j == (i+2))
	x4exact(i,j)=sqrt(((double)i*2.+1.)*((double)i*2.+2.)*((double)i*2.+3.)*((double)i*2.+4.));
      if (j == (i+1))
	x4exact(i,j)=(8.*(double)i+6.)*sqrt(((double)i*2.+1.)*((double)i*2.+2.));
      if (j == (i-1))
	x4exact(i,j)=(8.*(double)i-2.)*sqrt(((double)i*2.)*((double)i*2.-1.));
      if (j == (i-2))
	x4exact(i,j)=sqrt(((double)i*2.)*((double)i*2.-1.)*((double)i*2.-2.)*((double)i*2.-3.));
      x4exact(i,j)*=.25;
    }
  return x4exact;
}

matrix x4harmonic_A(int size)
  // for x4exact, change i from x4harmonic to 2i+1.
  // (j==(i+2)) here corresponds to (j==(i+4)) in x4harmonic, etc.
{
  matrix x4exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      x4exact(i,j)=0.;
      if (j == i)
	x4exact(i,j)=6.*((double)(2*i+1)*(double)(2*i+2))+3.;
      if (j == (i+2))
	x4exact(i,j)=sqrt(((double)i*2.+2.)*((double)i*2.+3.)*((double)i*2.+4.)*((double)i*2.+5.));
      if (j == (i+1))
	x4exact(i,j)=(8.*(double)i+10.)*sqrt(((double)i*2.+2.)*((double)i*2.+3.));
      if (j == (i-1))
	x4exact(i,j)=(8.*(double)i+2.)*sqrt(((double)i*2.+1)*((double)i*2.));
      if (j == (i-2))
	x4exact(i,j)=sqrt(((double)i*2.+1)*((double)i*2.)*((double)i*2.-1.)*((double)i*2.-2.));
      x4exact(i,j)*=.25;
    }
  return x4exact;
}

matrix pharmonic(int size)
{
  // actually -iP for harmonic oscillator
  matrix pexact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      pexact(i,j)=0.;
      if (j == (i+1))
	pexact(i,j)=-sqrt((double)i+1.); 
      if (j == (i-1))
	pexact(i,j)=sqrt((double)i);
      pexact(i,j)*=sqrt(.5);
    }
  return pexact;
}

matrix pharmonic_S(int size)
{
  // actually -iP_S for harmonic oscillator
  matrix pexact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      pexact(i,j)=0.;
      if (j == i)
	pexact(i,j)=sqrt((double)i*2.+1.); 
      if (j == (i+1))
	pexact(i,j)=-sqrt((double)i*2.+2.);
      pexact(i,j)*=sqrt(.5);
    }
  return pexact;
}

matrix pharmonic_A(int size)
{
  // actually -iP_A for harmonic oscillator
  matrix pexact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      pexact(i,j)=0.;
      if (j == i)
	pexact(i,j)=-sqrt((double)i*2.+1.); 
      if (j == (i-1))
	pexact(i,j)=sqrt((double)i*2.);
      pexact(i,j)*=sqrt(.5);
    }
  return pexact;
}

matrix p2harmonic(int size)
{
  // P^2 for harmonic oscillator
  matrix p2exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      p2exact(i,j)=0.;
      if (j == i)
	p2exact(i,j)=-2.*(double)i-1.;
      if (j == (i+2))
	p2exact(i,j)=sqrt(((double)i+1.)*((double)i+2.));
      if (j == (i-2))
	p2exact(i,j)=sqrt((double)i*((double)i-1.)); 
      p2exact(i,j)*=-.5;
    }
  return p2exact;
}

matrix p2harmonic_S(int size)
{
  matrix p2exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      p2exact(i,j)=0.;
      if (j == i)
	p2exact(i,j)=-4.*(double)i-1.;
      if (j == (i+1))
	p2exact(i,j)=sqrt(((double)i*2.+1.)*((double)i*2.+2.));
      if (j == (i-1))
	p2exact(i,j)=sqrt(((double)i*2.)*((double)i*2.-1.));
      p2exact(i,j)*=-.5;
    }
  return p2exact;
}

matrix p2harmonic_A(int size)
{
  matrix p2exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      p2exact(i,j)=0.;
      if (j == i)
	p2exact(i,j)=-4.*(double)i-3.;
      if (j == (i+1))
	p2exact(i,j)=sqrt(((double)i*2.+2.)*((double)i*2.+3.));
      if (j == (i-1))
	p2exact(i,j)=sqrt(((double)i*2.+1.)*((double)i*2.));
      p2exact(i,j)*=-.5;
    }
  return p2exact;
}

matrix p4harmonic(int size)
{
  matrix p4exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      p4exact(i,j)=0.;
      if (j == i)
	p4exact(i,j)=6.*((double)(i*i)+(double)i)+3.;
      if (j == (i+4))
	p4exact(i,j)=sqrt(((double)i+1.)*((double)i+2.)*((double)i+3.)*((double)i+4.));
      if (j == (i+2))
	p4exact(i,j)=-(4.*(double)i+6.)*sqrt(((double)i+1.)*((double)i+2.));
      if (j == (i-2))
	p4exact(i,j)=-(4.*(double)i-2.)*sqrt((double)i*((double)i-1.));
      if (j == (i-4))
	p4exact(i,j)=sqrt((double)i*((double)i-1.)*((double)i-2.)*((double)i-3.));
      p4exact(i,j)*=.25;
    }
  return p4exact;
}

matrix p4harmonic_S(int size)
{
  matrix p4exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      p4exact(i,j)=0.;
      if (j == i)
	p4exact(i,j)=12.*((double)(2*i*i)+(double)i)+3.;
      if (j == (i+2))
	p4exact(i,j)=sqrt(((double)i*2.+1.)*((double)i*2.+2.)*((double)i*2.+3.)*((double)i*2.+4.));
      if (j == (i+1))
	p4exact(i,j)=-(8.*(double)i+6.)*sqrt(((double)i*2.+1.)*((double)i*2.+2.));
      if (j == (i-1))
	p4exact(i,j)=-(8.*(double)i-2.)*sqrt(((double)i*2.)*((double)i*2.-1.));
      if (j == (i-2))
	p4exact(i,j)=sqrt(((double)i*2.)*((double)i*2.-1.)*((double)i*2.-2.)*((double)i*2.-3.));
      p4exact(i,j)*=.25;
    }
  return p4exact;
}

matrix p4harmonic_A(int size)
{
  matrix p4exact(size,size);
  for (int i=0;i<size;i++)
    for (int j=0;j<size;j++) {
      p4exact(i,j)=0.;
      if (j == i)
	p4exact(i,j)=6.*((double)(2*i+1)*(double)(2*i+2))+3.;
      if (j == (i+2))
	p4exact(i,j)=sqrt(((double)i*2.+2.)*((double)i*2.+3.)*((double)i*2.+4.)*((double)i*2.+5.));
      if (j == (i+1))
	p4exact(i,j)=-(8.*(double)i+10.)*sqrt(((double)i*2.+2.)*((double)i*2.+3.));
      if (j == (i-1))
	p4exact(i,j)=-(8.*(double)i+2.)*sqrt(((double)i*2.+1.)*((double)i*2.));
      if (j == (i-2))
	p4exact(i,j)=sqrt(((double)i*2.+1.)*((double)i*2.)*((double)i*2.-1.)*((double)i*2.-2.));
      p4exact(i,j)*=.25;
    }
  return p4exact;
}

diagmat harmonicoscillator(int size)
{
  diagmat H0(size);
  for (int i=0;i<size;i++)
    H0(i)=(double)i+.5;
  return H0;
}

diagmat harmonicoscillator_S(int size)
{
  diagmat H0(size);
  for (int i=0;i<size;i++)
    H0(i)=(double)i*2.+.5;
  return H0;
}

diagmat harmonicoscillator_A(int size)
{
  diagmat H0(size);
  for (int i=0;i<size;i++)
    H0(i)=(double)i*2.+1.5;
  return H0;
}

matrix heaviside(vector X,int size,int leftflag)
{
  // given the position operator X in the position representation
  // i.e. discrete value representation (DVR)
  // find left and right projection operators P (heaviside function)
  // when leftflag=0, the left projection is obtained
  matrix P(size,size);
  if (leftflag==0) {
    for (int i=0;i<size;i++) {
      if (X(i)<0.) P(i,i)=1.;
    }
  }
  else {
    for (int i=0;i<size;i++) {
      if (X(i)>0.) P(i,i)=1.;
    }
  }
  for (int i=0;i<size;i++) {
    if (fabs(X(i))<1.e-15) P(i,i)=.5;
  }
  return P;
}

// pn hacks
matrix point_x(vector X,int size,int index_x)
{
  // given the position operator X in the position representation
  // i.e. discrete value representation (DVR)
  matrix P(size,size);
  P(index_x,index_x)=1.;
  return P;
}
// end pn hacks

cvector QDOdiag(cmatrix Proj,cmatrix A,cvector Ep,cdiagmat expbetaEp,cmatrix L,cmatrix R,double beta,int size)
{
  // in the centroid formulation, a symmetrized property A by Proj is calculated
  // given diagonalized exp(-beta H') and its left and right matrices (L and R)
  // i.e., L H' R = D (D: some diagonal matrix)
  // returns only the diagonal elements of the fourier representation of centroid property A
  // if non-diagonal elements are needed, use "QDO" function
  cvector qdoA(size);
  cmatrix Projprime=(L*Proj)*R;
  cmatrix Aprime=(L*A)*R;
  for (int i=0;i<size;i++) {
    qdoA(i)=complex(0.,0.);
    for (int k=0;k<size;k++) {
      complex term;
      if (k==i) {
	term=expbetaEp(i)*Projprime(i,k)*Aprime(k,i);
      }
      else {
	complex Ediff=Ep(i)-Ep(k);
	complex expdiff=expbetaEp(k)-expbetaEp(i);
	term=Projprime(i,k)*Aprime(k,i)*expdiff/(complex(beta,0.)*Ediff);
      }
      qdoA(i)+=term;
    }
  }
  return qdoA;
}

cmatrix QDO(cmatrix Proj,cmatrix A,cvector Ep,cdiagmat expbetaEp,cmatrix L,cmatrix R,double beta,int size)
{
  cmatrix qdoA(size,size);
  cmatrix Projprime=(L*Proj)*R;
  cmatrix Aprime=(L*A)*R;
  cmatrix Ediff(size,size);
  cmatrix expdiff(size,size);
  for (int i=0;i<size;i++) {
    for (int k=0;k<size;k++) {
      Ediff(i,k)=Ep(i)-Ep(k);
      expdiff(i,k)=expbetaEp(k)-expbetaEp(i);
    }
  }
  for (int i=0;i<size;i++) {
    for (int j=0;j<size;j++) {
      qdoA(i,j)=complex(0.,0.);
      for (int k=0;k<size;k++) {
	complex term;
	if (k==i) {
	  term=expbetaEp(i)*Projprime(i,k)*Aprime(k,j);
	}
	else {
	  term=Projprime(i,k)*Aprime(k,j)*expdiff(i,k)/(complex(beta,0.)*Ediff(i,k));
	}
	qdoA(i,j)+=term;
      }
    }
  }
  return qdoA;
}

cvector QDOtimediag(cmatrix Proj,cmatrix A,cvector Ep,cdiagmat expbetaEp,cmatrix L,cmatrix R,vector E,matrix eigenvecs,matrix trnspeigenvecs,matrix zeromat,double beta,double t,int size)
{
  // returns only the diagonal elements of the fourier representation centroid property A at time t
  // if non-diagonal elements are needed, use "QDOtime" function
  cvector qdoAt(size);
  cmatrix Projprime=(L*Proj)*R;
  cmatrix Aprime=(L*A)*R;
  cdiagmat iE(size);
  for (int i=0;i<size;i++) {
    iE(i)=complex(0.,E(i));
  }
  cdiagmat expitE=expomatrix(iE,t,size);
  cmatrix expitH=(complexm(eigenvecs,zeromat)*expitE)*complexm(trnspeigenvecs,zeromat);
  //cdiagmat expnegitE=expomatrix(iE,-t,size);
  //cmatrix expnegitH=(complexm(eigenvecs,zeromat)*expnegitE)*complexm(trnspeigenvecs,zeromat);
  cmatrix expnegitH=conjugate(expitH,size,size);
  expitH=(L*expitH)*R;
  expnegitH=(L*expnegitH)*R;
  cmatrix Ediff(size,size);
  cmatrix expdiff(size,size);
  for (int l=0;l<size;l++) {
    for (int m=0;m<size;m++) {
      Ediff(l,m)=Ep(l)-Ep(m);
      expdiff(l,m)=expbetaEp(m)-expbetaEp(l);
    }
  }
  for (int i=0;i<size;i++) {
    qdoAt(i)=complex(0.,0.);
    for (int k=0;k<size;k++) {
      for (int l=0;l<size;l++) {
	for (int m=0;m<size;m++) {
	  complex term;
	  if (l==m) {
	    term=expnegitH(i,l)*expitH(m,k)*Projprime(l,m)*Aprime(k,i)*expbetaEp(l);
	  }
	  else {
	    term=expnegitH(i,l)*expitH(m,k)*Projprime(l,m)*Aprime(k,i)*expdiff(l,m)/(Ediff(l,m)*complex(beta,0.));
	  }
	  qdoAt(i)+=term;
	}
      }
    }
  }
  return qdoAt;
}

cmatrix QDOtime(cmatrix Proj,cmatrix A,cvector Ep,cdiagmat expbetaEp,cmatrix L,cmatrix R,vector E,matrix eigenvecs,matrix trnspeigenvecs,matrix zeromat,double beta,double t,int size)
{
  cmatrix qdoAt(size,size);
  cmatrix Projprime=(L*Proj)*R;
  cmatrix Aprime=(L*A)*R;
  cdiagmat iE(size);
  for (int i=0;i<size;i++) {
    iE(i)=complex(0.,E(i));
  }
  cdiagmat expitE=expomatrix(iE,t,size);
  cmatrix expitH=(complexm(eigenvecs,zeromat)*expitE)*complexm(trnspeigenvecs,zeromat);
  //cdiagmat expnegitE=expomatrix(iE,-t,size);
  //cmatrix expnegitH=(complexm(eigenvecs,zeromat)*expnegitE)*complexm(trnspeigenvecs,zeromat);
  cmatrix expnegitH=conjugate(expitH,size,size);
  expitH=(L*expitH)*R;
  expnegitH=(L*expnegitH)*R;
  cmatrix Ediff(size,size);
  cmatrix expdiff(size,size);
  for (int l=0;l<size;l++) {
    for (int m=0;m<size;m++) {
      Ediff(l,m)=Ep(l)-Ep(m);
      expdiff(l,m)=expbetaEp(m)-expbetaEp(l);
    }
  }
  for (int i=0;i<size;i++) {
    for (int j=0;j<size;j++) {
      qdoAt(i,j)=complex(0.,0.);
      for (int k=0;k<size;k++) {
	for (int l=0;l<size;l++) {
	  for (int m=0;m<size;m++) {
	    complex term;
	    if (l==m) {
	      term=expnegitH(i,l)*expitH(m,k)*Projprime(l,m)*Aprime(k,j)*expbetaEp(l);
	    }
	    else {
	      term=expnegitH(i,l)*expitH(m,k)*Projprime(l,m)*Aprime(k,j)*expdiff(l,m)/(Ediff(l,m)*complex(beta,0.));
	    }
	    qdoAt(i,j)+=term;
	  }
	}
      }
    }
  }
  return qdoAt;
}

cmatrix timeoperator(vector E,matrix P,matrix trnspsP,cmatrix A,matrix zeromat,double t,int size)
{
  // returns the operator at time t A(t), given the operator A
  // A(t) = exp(i t H) A exp(-i t H)
  // H (hamiltonian) and A are expressed in the same basis
  // where trnspsP H P = E
  cdiagmat itE(size);
  for (int i=0;i<size;i++) {
    itE(i)=complex(0.,t*E(i));
  }
  cdiagmat expitE=expomatrix(itE,1.,size);
  cmatrix expitH=(complexm(P,zeromat)*expitE)*complexm(trnspsP,zeromat);
  cmatrix expnegitH=conjugate(expitH,size,size);
  cmatrix Aoft=(expitH*A)*expnegitH;
  return Aoft;
}

cmatrix FourierQDO(cvector Ep,cdiagmat expbetaEp,cmatrix L,cmatrix R,cmatrix Proj,double beta,int size)
{
  // returns Fourier representation of unnormalized symmetry adapted QDO
  // with respect to HO eigenstate basis
  // input symmetry or projection operator with respect to HO eigenstate basis
  cmatrix expbetaHp_Proj(size,size);
  cmatrix Projprime=(L*Proj)*R;
  for (int i=0;i<size;i++) {
    for (int j=0;j<size;j++) {
      if (i==j) {
	expbetaHp_Proj(i,j)=Projprime(i,j)*expbetaEp(i);
      }
      else {
	complex Ediff=Ep(i)-Ep(j);
	complex expdiff=expbetaEp(j)-expbetaEp(i);
	expbetaHp_Proj(i,j)=Projprime(i,j)*expdiff/(complex(-beta,0.)*Ediff);
      }
    }
  }
  expbetaHp_Proj=(R*expbetaHp_Proj)*L;
  return expbetaHp_Proj;
}

matrix orderbasis(int size)
  // to order basis (B to b) such that
  // basis elements in the odd positions first, then basis elements in the even positions
  // (i.e., a special kind of permutation matrix)
  // computes the transition matrix P from b to B
  // matrix(in b) = P^(-1) * matrix(in B) * P; P = P^(-1)
  // e.g. xharmonic_SA=invorderbasis * xharmonic * orderbasis
{
  matrix trans(size,size);
  int halfsize;
  if (size%2 == 0) halfsize=size/2;
  else halfsize=(size+1)/2;
  for (int i=0;i<size;i++) {
    for (int j=0;j<halfsize;j++) {
      trans(i,j)=0.;
      if (i == 2*j)
	trans(i,j)=1.;
    }
    for (int j=halfsize;j<size;j++) {
      trans(i,j)=0.;
      if (i == 2*(j-halfsize)+1)
	trans(i,j)=1.;
    }
  }
  return trans;
}

matrix invorderbasis(int size)
  // computes the inverse of orderbasis P^(-1)
{
  matrix trans(size,size);
  int halfsize;
  if (size%2 == 0) halfsize=size/2;
  else halfsize=(size+1)/2;
  for (int j=0;j<size;j++) {
    for (int i=0;i<halfsize;i++) {
      trans(i,j)=0.;
      if (j == 2*i)
	trans(i,j)=1.;
    }
    for (int i=halfsize;i<size;i++) {
      trans(i,j)=0.;
      if (j == 2*(i-halfsize)+1)
	trans(i,j)=1.;
    }
  }
  return trans;
}

void outputmatrix(matrix A,int size1,int size2)
{
  double A_ij;
  for (int i=0;i<size1;i++) {
    for (int j=0;j<size2;j++) {
      A_ij=A(i,j);
      if (fabs(A_ij)<1.e-15) A_ij=0.;
      cout<<A_ij<<" ";
    }
    cout<<endl;
  }
  return;
}

void outputvector(vector A, int size)
{
  double A_i;
  for (int i=0;i<size;i++) {
    A_i=A(i);
    if (fabs(A_i)<1.e-15) A_i=0.;
    cout<<A_i<<endl;
  }
  return;
}

matrix zeromatrix(int size1,int size2)
  // zero matrix
{
  matrix A(size1,size2);
  for (int i=0;i<size1;i++) {
    for (int j=0;j<size2;j++) {
      A(i,j)=0.;
    }
  }
  return A;
}

cmatrix czeromatrix(int size1,int size2)
{
  cmatrix A(size1,size2);
  for (int i=0;i<size1;i++) {
    for (int j=0;j<size2;j++) {
      A(i,j)=complex(0.,0.);
    }
  }
  return A;
}

matrix identitymatrix(int size)
  // identity matrix
{
  matrix A(size,size);
  for (int i=0;i<size;i++) {
    for (int j=0;j<size;j++) {
      A(i,j)=delta(i,j);
    }
  }
  return A;
}

matrix symprojmatrix(int size,int halfsize)
// symmetric projection matrix
{
  matrix S(size,size);
  for (int i=0;i<size;i++) {
    for (int j=0;j<size;j++) {
      S(i,j)=0.;
      if (i<halfsize && j<halfsize) S(i,j)=delta(i,j);
    }
  }
  return S;
}

matrix antisymprojmatrix(int size,int halfsize)
// antisymmetric projection matrix
{
  matrix A(size,size);
  for (int i=0;i<size;i++) {
    for (int j=0;j<size;j++) {
      A(i,j)=0.;
      if (i>=halfsize && j>=halfsize) A(i,j)=delta(i,j);
    }
  }
  return A;
}

cmatrix pickelement(cmatrix A,int k,int kp,int size)
  // returns matrix elements from some part of another big matrix
{
  cmatrix partA(size,size);
  for (int i=0;i<size;i++) {
    for (int j=0;j<size;j++) {
      partA(i,j)=A(k*size+i,kp*size+j);
    }
  }
  return partA;
}

void putelement(cmatrix& A,cmatrix partA,int k,int kp,int size)
  // writes some part of a big matrix using an input matrix
{
  for (int i=0;i<size;i++) {
    for (int j=0;j<size;j++) {
      A(k*size+i,kp*size+j)=partA(i,j);
    }
  }
}

cdiagmat diagelement(cmatrix A,cmatrix B,int size)
  // computes only diagonal elements resulting from a matrix multiplication
  // i.e., diagonal elements of (A * B)
  // for now, square matrices only
{
  cdiagmat Adiag(size);
  for (int i=0;i<size;i++) {
    Adiag(i)=complex(0.,0.);
    for (int j=0;j<size;j++) {
      Adiag(i)+=(A(i,j)*B(j,i));
    }
  }
  return Adiag;
}

matrix totmat(int size,int halfsize,matrix A1,matrix A2,matrix A3,matrix A4)
  // put matrices together to get the total (size x size) matrix
  // A = [A1 A2],[A3 A4]
{
  matrix A(size,size);
  for (int i=0;i<size;i++) {
    for (int j=0;j<size;j++) {
      if (i<halfsize && j<halfsize) {
	A(i,j)=A1(i,j);
      }
      if (i>=halfsize && j>=halfsize) {
	A(i,j)=A4(i-halfsize,j-halfsize);
      }
      if (i<halfsize && j>=halfsize) {
	A(i,j)=A2(i,j-halfsize);
      }
      if (i>=halfsize && j<halfsize) {
	A(i,j)=A3(i-halfsize,j);
      }
    }
  }
  return A;
}

cmatrix adjoint(cmatrix A,int rowsize,int colsize)
  // adjoint of a complex matrix of dimension (rowsize,colsize)
  // i.e. complex conjugate and transpose
{
  cmatrix adjA(colsize,rowsize);
  for (int i=0;i<colsize;i++) {
    for (int j=0;j<rowsize;j++) {
      adjA(i,j)=complex(real(A(j,i)),-imag(A(j,i)));
    }
  }
  return adjA;
}

cmatrix conjugate(cmatrix A,int rowsize,int colsize)
{
  // complex conjugate of a complex matrix
  cmatrix conjA(rowsize,colsize);
  for (int i=0;i<rowsize;i++) {
    for (int j=0;j<colsize;j++) {
      conjA(i,j)=complex(real(A(i,j)),-imag(A(i,j)));
    }
  }
  return conjA;
}

cdiagmat conjugate(cdiagmat A,int size)
{
  // complex conjugate of a complex diagonal matrix
  cdiagmat conjA(size);
  for (int i=0;i<size;i++) {
    conjA(i)=complex(real(A(i)),-imag(A(i)));
  }
  return conjA;
}

cvector conjugate(cvector A,int size)
{
  // complex conjugate of a complex vector
  cvector conjA(size);
  for (int i=0;i<size;i++) {
    conjA(i)=complex(real(A(i)),-imag(A(i)));
  }
  return conjA;
}

double delta(int i,int ip)
  // Kronecker Delta function
{
  if (i == ip) return 1.;
  else return 0.;
}

complex partialtrace(cmatrix A,int size,int flag)
  // sum of alternating diagonal elements
{
  complex tr(0.,0.);
  if (flag==0)
    for (int i=0;i<size;i+=2) tr+=A(i,i);
  if (flag==1)
    for (int i=1;i<size;i+=2) tr+=A(i,i);
  return tr;
}

double partialtrace(matrix A,int size,int flag)
  // sum of alternating diagonal elements
{
  double tr=0.;
  if (flag==0)
    for (int i=0;i<size;i+=2) tr+=A(i,i);
  if (flag==1)
    for (int i=1;i<size;i+=2) tr+=A(i,i);
  return tr;
}

complex computetrace(cmatrix A,int start,int finish)
  // trace of a complex matrix
{
  complex tr(0.,0.);
  for (int i=start;i<finish;i++) tr+=A(i,i);
  return tr;
}

double computetrace(matrix A,int start,int finish)
  // trace of a real matrix
{
  double tr=0.;
  for (int i=start;i<finish;i++) tr+=A(i,i);
  return tr;
}

complex computetrace(cdiagmat A,int start,int finish)
  // trace of a complex diagonal matrix
{
  complex tr(0.,0.);
  for (int i=start;i<finish;i++) tr+=A(i);
  return tr;
}

double computetrace(diagmat A,int start,int finish)
  // trace of a real diagonal matrix
{
  double tr=0.;
  for (int i=start;i<finish;i++) tr+=A(i);
  return tr;
}

complex computesum(cvector A,int start,int finish)
  // sum of complex vector elements
  // useful for sum of eigenvalues (in a vector form), i.e., trace of a diagonal matrix
{
  complex sum(0.,0.);
  for (int i=start;i<finish;i++) sum+=A(i);
  return sum;
}

double computesum(vector A,int start,int finish)
  // sum of real vector elements
{
  double sum=0.;
  for (int i=start;i<finish;i++) sum+=A(i);
  return sum;
}

cmatrix expomatrix(cmatrix A,cvector& D,cdiagmat& expD,cmatrix& L,cmatrix& R,double f,int size,int flag)
  // exponentiate a complex matrix
  // flag = 0 is for hermitian matrice
  // assign flag = 1 for general (non-symmetric) real matrices
{
  D=diag(A,L,R);
  if (flag==0) L=adjoint(R,size,size);
  else L=inverse(R);
  expD=expovec(D,f,size);
  cmatrix expA=(R*expD)*L;
  return expA;
}

matrix expomatrix(matrix& A,vector& D,diagmat& expD,matrix& invA,double f,int size,int flag)
  // exponentiate a real matrix
  // flag = 0 is for symmetric matrice
  // assign flag = 1 for general (non-symmetric) real matrices
{
  D=diag(A);
  if (flag==0) invA=transpose(A);
  else invA=inverse(A);
  expD=expovec(D,f,size);
  matrix expA=(A*expD)*invA;
  return expA;
}

cdiagmat expomatrix(cdiagmat A,double f,int size)
  // exponentiate a complex diagonal matrix
{
  cdiagmat expA(size);
  for (int i=0;i<size;i++) {
    //double phase=f*imag(A(i));
    //expA(i)=complex(exp(f*real(A(i))),0.)*complex(cos(phase),sin(phase));
    expA(i)=exp(complex(f,0.)*A(i));
  }
  return expA;
}

diagmat expomatrix(diagmat A,double f,int size)
  // exponentiates a real diagonal matrix
{
  diagmat expA(size);
  for (int i=0;i<size;i++)
    expA(i)=exp(f*A(i));
  return expA;
}

cdiagmat expovec(cvector A,double f,int size)
  // exponentiates a complex vector
  // and returns a complex diagonal matrix
{
  cdiagmat expA(size);
  for (int i=0;i<size;i++) {
    //double phase=f*imag(A(i));
    //expA(i)=complex(exp(f*real(A(i))),0.)*complex(cos(phase),sin(phase));
    expA(i)=exp(complex(f,0.)*A(i));
  }
  return expA;
}

diagmat expovec(vector A,double f,int size)
  // exponentiates a real vector form
  // and returns a real diagonal matrix
{
  diagmat expA(size);
  for (int i=0;i<size;i++)
    expA(i)=exp(f*A(i));
  return expA;
}

complex exptrace(cmatrix A,double f,int start,int finish,int size,int flag)
  // trace of an exponential of a complex matrix
{
  cvector D(size);
  cdiagmat expD(size);
  cmatrix L(size,size);
  cmatrix R(size,size);
  cmatrix expA=expomatrix(A,D,expD,L,R,f,size,flag);
  complex tr=computetrace(expA,start,finish);
  return tr;
}

double exptrace(matrix A,double f,int start,int finish,int size,int flag)
  // trace of an exponential of a real matrix
{
  vector D(size);
  diagmat expD(size);
  matrix invA(size,size);
  matrix expA=expomatrix(A,D,expD,invA,f,size,flag);
  double tr=computetrace(expA,start,finish);
  return tr;
}

complex exptrace(cdiagmat A,double f,int start,int finish,int size)
  // trace of exponential of a complex diagonal matrix A with a real factor f
  // i.e., Tr{ exp(f*A) }
{
  cdiagmat expA=expomatrix(A,f,size);
  complex tr=computetrace(expA,start,finish);
  return tr;
}

double exptrace(diagmat A,double f,int start,int finish,int size)
  // trace of exponential of a real diagonal matrix
{
  diagmat expA=expomatrix(A,f,size);
  double tr=computetrace(expA,start,finish);
  return tr;
}

complex expsum(cvector A,double f,int start,int finish,int size)
  // sum of exponentials of complex vector elements
{
  cdiagmat expA=expovec(A,f,size);
  complex sum=computetrace(expA,start,finish);
  return sum;
}

double expsum(vector A,double f,int start,int finish,int size)
  // sum of exponentials of real vector elements
{
  diagmat expA=expovec(A,f,size);
  double sum=computetrace(expA,start,finish);
  return sum;
}

cvector singleFT(cvector A,double kmax,double xmax,int nk)
  // returns the following function f(x)
  // f(x) = integral (dk/(2*M_PI)) exp(i k x) F(k)
{
  cvector f(nk);
  double dk=2.*kmax/(double)nk;
  double dx=2.*xmax/(double)nk;
  for (int i=0;i<nk;i++) {
    double xvalue=-xmax+dx*(double)i;
    for (int k=0;k<nk;k++) {
      double kvalue=-kmax+dk*(double)k;
      double phase=kvalue*xvalue;
      //complex exphase=complex(cos(phase),sin(phase));
      complex exphase=exp(complex(0.,phase));
      f(i)+=exphase*A(k);
    }
    f(i)*=dk/(2.*M_PI);
  }
  return f;
}

cmatrix doubleFT_forward(cmatrix A,double kmax,double kpmax,double xmax,double pmax,int nk,int nkp,int nx,int np)
  // returns the following function f(x,p)
  // f(x,p) = double integral dk dkp exp(i k x) exp (i kp p) F(k,kp)
{
  cmatrix f(nx,np);
  /*
  double dk=2.*kmax/(double)nk;
  double dkp=2.*kpmax/(double)nkp;
  double dx=2.*xmax/(double)nk;
  double dp=2.*pmax/(double)nkp;
  */
  double dk=2.*kmax/((double)(nk-1));
  double dkp=2.*kpmax/((double)(nkp-1));
  double dx=2.*xmax/((double)(nx-1));
  double dp=2.*pmax/((double)(np-1));

  for (int i=0;i<nx;i++) {
    double xvalue=-xmax+dx*(double)i;
    for (int j=0;j<np;j++) {
      double pvalue=-pmax+dp*(double)j;
      for (int k=0;k<nk;k++) {
	double kvalue=-kmax+dk*(double)k;
	double phasex=kvalue*xvalue;

	//complex exphase=complex(cos(phasex),sin(phasex));
	complex exphase=exp(complex(0.,phasex));
	for (int kp=0;kp<nkp;kp++) {
	  double kpvalue=-kpmax+dkp*(double)kp;
	  double phasep=kpvalue*pvalue;

	  //complex epphase=complex(cos(phasep),sin(phasep));
	  complex epphase=exp(complex(0.,phasep));
	  f(i,j)+=exphase*epphase*A(k,kp);
	}
      }
    }
  }
  return f;
}

cmatrix doubleFT_inverse(cmatrix A,double kmax,double kpmax,double xmax,double pmax,int nk,int nkp,int nx,int np)
  // returns the following function f(x,p)
  // f(x,p) = (1/2*M_PI) double integral dk dkp exp(-i k x) exp (-i kp p)
  //                                            F(k,kp)
{
  cmatrix f(nx,np);
  /*
  double dk=2.*kmax/(double)nk;
  double dkp=2.*kpmax/(double)nkp;
  double dx=2.*xmax/(double)nk;
  double dp=2.*pmax/(double)nkp;
  */
  double dk=2.*kmax/((double)(nk-1));
  double dkp=2.*kpmax/((double)(nkp-1));
  double dx=2.*xmax/((double)(nx-1));
  double dp=2.*pmax/((double)(np-1));

  complex factor(dk*dkp/(2.*M_PI),0.);

  for (int i=0;i<nx;i++) {
    double xvalue=-xmax+dx*(double)i;
    for (int j=0;j<np;j++) {
      double pvalue=-pmax+dp*(double)j;
      for (int k=0;k<nk;k++) {
	double kvalue=-kmax+dk*(double)k;
	double phasex=-kvalue*xvalue;

	//complex exphase=complex(cos(phasex),sin(phasex));
	complex exphase=exp(complex(0.,phasex));
	for (int kp=0;kp<nkp;kp++) {
	  double kpvalue=-kpmax+dkp*(double)kp;
	  double phasep=-kpvalue*pvalue;

	  //complex epphase=complex(cos(phasep),sin(phasep));
	  complex epphase=exp(complex(0.,phasep));
	  f(i,j)+=exphase*epphase*A(k,kp);
	}
      }
      f(i,j)=factor*f(i,j);
    }
  }
  return f;
}

complex doubleFTpoint_forward(cmatrix A,double kmax,double kpmax,double xvalue,double pvalue,int nk,int nkp)
  // returns the following function f(xvalue,pvalue)
  // f(x,p) = double integral dk dkp exp(i k x) exp (i kp p) F(k,kp)
{
  complex f;
  /*
  double dk=2.*kmax/(double)nk;
  double dkp=2.*kpmax/(double)nkp;
  */
  double dk=2.*kmax/((double)(nk-1));
  double dkp=2.*kpmax/((double)(nkp-1));

  for (int k=0;k<nk;k++) {
    double kvalue=-kmax+dk*(double)k;

    // temp: for inverse FT, i.e., xvalue --> -xvalue
    double phasex=kvalue*xvalue;
    //double phasex=-kvalue*xvalue;
    // end temp

    //complex exphase=complex(cos(phasex),sin(phasex));
    complex exphase=exp(complex(0.,phasex));
    for (int kp=0;kp<nkp;kp++) {
      double kpvalue=-kpmax+dkp*(double)kp;

      // temp: for inverse FT
      double phasep=kpvalue*pvalue;
      //double phasep=-kpvalue*pvalue;
      // end temp

      //complex epphase=complex(cos(phasep),sin(phasep));
      complex epphase=exp(complex(0.,phasep));
      f+=exphase*epphase*A(k,kp);
    }
  }
  return f;
}

complex doubleFTpoint_inverse(cmatrix A,double kmax,double kpmax,double xvalue,double pvalue,int nk,int nkp)
  // returns the following function f(xvalue,pvalue)
  // f(x,p) = (1/2*M_PI) double integral dk dkp exp(-i k x) exp (-i kp p)
  //                                            F(k,kp)
{
  complex f;
  /*
  double dk=2.*kmax/(double)nk;
  double dkp=2.*kpmax/(double)nkp;
  */
  double dk=2.*kmax/((double)(nk-1));
  double dkp=2.*kpmax/((double)(nkp-1));

  complex factor(dk*dkp/(2.*M_PI),0.);

  for (int k=0;k<nk;k++) {
    double kvalue=-kmax+dk*(double)k;
    double phasex=-kvalue*xvalue;

    //complex exphase=complex(cos(phasex),sin(phasex));
    complex exphase=exp(complex(0.,phasex));
    for (int kp=0;kp<nkp;kp++) {
      double kpvalue=-kpmax+dkp*(double)kp;
      double phasep=-kpvalue*pvalue;

      //complex epphase=complex(cos(phasep),sin(phasep));
      complex epphase=exp(complex(0.,phasep));
      f+=exphase*epphase*A(k,kp);
    }
  }
  f=factor*f;
  return f;
}

void specialFT_centroid(int nk,int nkp,int ngridxc,int ngridpc,double dk,double dkp,vector kvalue,vector kpvalue,matrix frho,matrix frho_E,matrix ff,matrix ff_E,matrix ff_old_E,matrix fh,matrix fh_E,matrix fv,matrix fv_E,vector grho,vector grho_E,vector hrho,vector hrho_E,vector& gridxc,vector& gridpc,matrix& rhoc,matrix& rhoc_E,matrix& fc,matrix& fc_E,matrix& fc_old_E,matrix& hc,matrix& hc_E,matrix& vc,matrix& vc_E,vector& rhocg,vector& rhocg_E,vector& rhoch,vector& rhoch_E)
  // performs special 2D-FT for centroid density and centroid force
  // frho is real, and identical for all quadrants
  // ff is imaginary, and only different by sign for all quadrants
  // restricted to the first quadrants
  // for now, we keep it less general for centroid calculation
  // can be generalized to 2D-FT for functions with special properties
  // e.g. real, imaginary, symmetric about the axes/origin,...
{
  for (int j=0;j<ngridxc;j++) {
    for (int jp=0;jp<ngridpc;jp++) {
      rhoc(j,jp)=0.;
      rhoc_E(j,jp)=0.;
      fc(j,jp)=0.;
      fc_E(j,jp)=0.;

      // temp
      fc_old_E(j,jp)=0.;
      // end temp

      hc(j,jp)=0.;
      hc_E(j,jp)=0.;
      vc(j,jp)=0.;
      vc_E(j,jp)=0.;
      if (jp==0) {
	rhocg(j)=0.;
	rhocg_E(j)=0.;
      }
      if (j==0) {
	rhoch(jp)=0.;
	rhoch_E(jp)=0.;
      }
      for (int k=0;k<nk;k++) {
	double phasex=-kvalue(k)*gridxc(j);
	double cosphasex=cos(phasex);
	double sinphasex=sin(phasex);

	for (int kp=0;kp<nkp;kp++) {
	  double phasep=-kpvalue(kp)*gridpc(jp);
	  double cosphasep=cos(phasep);
	  double sinphasep=sin(phasep);

	  if ((k==0) && (kp==0)) {
	    rhoc(j,jp)+=.25*frho(k,kp);
	    rhoc_E(j,jp)+=.25*frho_E(k,kp);
	    hc(j,jp)+=.25*fh(k,kp);
	    hc_E(j,jp)+=.25*fh_E(k,kp);
	    vc(j,jp)+=.25*fv(k,kp);
	    vc_E(j,jp)+=.25*fv_E(k,kp);
	  }
	  else {
	    if ((k==0) || (kp==0)) {
	      if (k==0) {
		rhoc(j,jp)+=.5*cosphasep*frho(k,kp);
		rhoc_E(j,jp)+=.5*cosphasep*frho_E(k,kp);
		hc(j,jp)+=.5*cosphasep*fh(k,kp);
		hc_E(j,jp)+=.5*cosphasep*fh_E(k,kp);
		vc(j,jp)+=.5*cosphasep*fv(k,kp);
		vc_E(j,jp)+=.5*cosphasep*fv_E(k,kp);
	      }
	      else {
		rhoc(j,jp)+=.5*cosphasex*frho(k,kp);
		rhoc_E(j,jp)+=.5*cosphasex*frho_E(k,kp);
		hc(j,jp)+=.5*cosphasex*fh(k,kp);
		hc_E(j,jp)+=.5*cosphasex*fh_E(k,kp);
		vc(j,jp)+=.5*cosphasex*fv(k,kp);
		vc_E(j,jp)+=.5*cosphasex*fv_E(k,kp);
	      }
	    }
	    else {
	       rhoc(j,jp)+=cosphasex*cosphasep*frho(k,kp);
	       rhoc_E(j,jp)+=cosphasex*cosphasep*frho_E(k,kp);
	       hc(j,jp)+=cosphasex*cosphasep*fh(k,kp);
	       hc_E(j,jp)+=cosphasex*cosphasep*fh_E(k,kp);
	       vc(j,jp)+=cosphasex*cosphasep*fv(k,kp);
	       vc_E(j,jp)+=cosphasex*cosphasep*fv_E(k,kp);
	    }
	  }

	  if (k!=0) {
	    if (kp==0) {
	      fc(j,jp)+=.5*sinphasex*ff(k,kp);
	      fc_E(j,jp)+=.5*sinphasex*ff_E(k,kp);
	    }
	    else {
	      fc(j,jp)+=sinphasex*cosphasep*ff(k,kp);
	      fc_E(j,jp)+=sinphasex*cosphasep*ff_E(k,kp);
	    }
	  }

	  // temp
	  if (kp!=0) {
	    if (k==0) {
	      fc_old_E(j,jp)+=.5*sinphasep*ff_old_E(k,kp);
	    }
	    else {
	      fc_old_E(j,jp)+=cosphasex*sinphasep*ff_old_E(k,kp);
	    }
	  }
	  // end temp

	  if ((jp==0) && (kp==0)) {
	    if (k==0) {
	      rhocg(j)+=.5*grho(k);
	      rhocg_E(j)+=.5*grho_E(k);
	    }
	    else {
	      rhocg(j)+=cosphasex*grho(k);
	      rhocg_E(j)+=cosphasex*grho_E(k);
	    }
	  }
	  if ((j==0) && (k==0)) {
	    if (kp==0) {
	      rhoch(jp)+=.5*hrho(kp);
	      rhoch_E(jp)+=.5*hrho_E(kp);
	    }
	    else {
	      rhoch(jp)+=cosphasep*hrho(kp);
	      rhoch_E(jp)+=cosphasep*hrho_E(kp);
	    }
	  }
	}
      }
    }
  }
  // rhoc is multiplied by 4*dk*dkp/(2*M_PI)
  // where 4 comes from the integration over 4 quadrants
  // For a free particle, rho should be multiplied by sqrt(2*M_PI*beta*hbar^2/m) (not done here)
  double factor=2.*dk*dkp/M_PI;
  rhoc=factor*rhoc;
  rhoc_E=factor*rhoc_E;
  hc=factor*hc;
  hc_E=factor*hc_E;
  vc=factor*vc;
  vc_E=factor*vc_E;

  // temp
  fc_old_E=factor*fc_old_E;
  // end temp

  // fc is multiplied by -4*dk*dkp/(2*M_PI*rhoc),
  // where 4 comes from the integration over 4 quadrants
  fc=(-factor)*fc;
  fc_E=(-factor)*fc_E;

  double factordk=2.*dk;
  double factordkp=2.*dkp;
  rhocg=factordk*rhocg;
  rhocg_E=factordk*rhocg_E;
  rhoch=factordkp*rhoch;
  rhoch_E=factordkp*rhoch_E;
  return;
}

void writematrix(matrix& A,int nx,int ny,int flag)
  // given the 1st quadrant, fill out the rest
  // flag=0 for symmetric for all quadrants, e.g. centroid density
  // flag=1, e.g. centroid force
{
  if (flag==0) {
    for (int j=0;j<nx;j++) {
      for (int jp=0;jp<ny;jp++) {
	if ((j!=0) || (jp!=0)) {
	  if ((j==0) || (jp==0)) {
	    if (j==0) {
	      A(nx-1,ny-1-jp)=A(nx-1,ny-1+jp);
	    }
	    else {
	      A(nx-1-j,ny-1)=A(nx-1+j,ny-1);
	    }
	  }
	  else {
	    A(nx-1-j,ny-1-jp)=A(nx-1+j,ny-1+jp);
	    A(nx-1-j,ny-1+jp)=A(nx-1+j,ny-1+jp);
	    A(nx-1+j,ny-1-jp)=A(nx-1+j,ny-1+jp);
	  }
	}
      }
    }
  }
  else {
    for (int j=0;j<nx;j++) {
      for (int jp=0;jp<ny;jp++) {
	if ((j!=0) || (jp!=0)) {
	  if ((j==0) || (jp==0)) {
	    if (j==0) {
	      A(nx-1,ny-1-jp)=A(nx-1,ny-1+jp);
	    }
	    else {
	      A(nx-1-j,ny-1)=-A(nx-1+j,ny-1);
	    }
	  }
	  else {
	    A(nx-1-j,ny-1-jp)=-A(nx-1+j,ny-1+jp);
	    A(nx-1-j,ny-1+jp)=-A(nx-1+j,ny-1+jp);
	    A(nx-1+j,ny-1-jp)=A(nx-1+j,ny-1+jp);
	  }
	}
      }
    }
  }
  return;
}

void euler(double m,double dt,double force,double& newx,double& newp)
{
  double oldx=newx;
  double oldp=newp;
  newx=oldx+(oldp*dt/m)+(force*dt*dt/m);
  newp=oldp+force*dt;
  return;
}

void rungekutta(double m,double dt,Interp2d f,double& newx,double& newp)
{
  double oldx=newx;
  double oldp=newp;

  // k1
  double k1_x=oldp/m;
  double k1_p=f.interp2d(oldx,oldp);
  double q_x=oldx+(.5*dt*k1_x);
  double q_p=oldp+(.5*dt*k1_p);

  // k2
  double k2_x=q_p/m;
  double k2_p=f.interp2d(q_x,q_p);
  q_x=oldx+(.5*dt*k2_x);
  q_p=oldp+(.5*dt*k2_p);

  // k3
  double k3_x=q_p/m;
  double k3_p=f.interp2d(q_x,q_p);
  q_x=oldx+(dt*k3_x);
  q_p=oldp+(dt*k3_p);

  // k4
  double k4_x=q_p/m;
  double k4_p=f.interp2d(q_x,q_p);

  newx+=(dt/6.)*(k1_x+2.*k2_x+2.*k3_x+k4_x);
  newp+=(dt/6.)*(k1_p+2.*k2_p+2.*k3_p+k4_p);
  return;
}

void corrfn0_DK(vector& corr,vector& corr_S,vector& corr_A,double Z,double Z_S,double Z_A,vector E,matrix A,double beta,int size,int ntimesteps,double starttime,double dt)
  // double-Kubo transformed zero time correlation function for
  // regular correlation function, and with symmetry
  // A must be in the same basis as D
  // < A A(t) >
{
  for (int i=0;i<ntimesteps;i++) {
    corr(i)=0.;
    corr_S(i)=0.;
    corr_A(i)=0.;
    double time=starttime+(double)i*dt;
    for (int j=0;j<size;j++) {
      double expbetaEj=exp(-beta*E(j));
      for (int k=0;k<size;k++) {
	double factor;
	if (j==k) {
	  factor=expbetaEj;
	}
	else {
	  double Ediff=E(j)-E(k);
	  factor=2.*(exp(-beta*E(k))-expbetaEj*(1.+beta*Ediff))*cos(Ediff*time)/pow(beta*Ediff,2.);
	}
	double corr_exact=A(j,k)*A(k,j)*factor;
	corr(i)+=corr_exact;
	if (j%2==0) {
	  corr_S(i)+=corr_exact;
	}
	else {
	  corr_A(i)+=corr_exact;
	}
      }
    }
  }
  corr=(1./Z)*corr;
  corr_S=(1./Z_S)*corr_S;
  corr_A=(1./Z_A)*corr_A;
  return;
}

void corrfn0_SK(vector& corr,vector& corr_S,vector& corr_A,double Z,double Z_S,double Z_A,vector E,matrix A,double beta,int size,int ntimesteps,double starttime,double dt)
  // single-Kubo transformed zero time correlation function for
  // regular correlation function, and with symmetry
  // A must be in the same basis as D
  // < A A(t) >
{
  for (int i=0;i<ntimesteps;i++) {
    corr(i)=0.;
    corr_S(i)=0.;
    corr_A(i)=0.;
    double time=starttime+(double)i*dt;
    for (int j=0;j<size;j++) {
      double expbetaEj=exp(-beta*E(j));
      for (int k=0;k<size;k++) {
	double factor;
	if (j==k) {
	  factor=expbetaEj;
	}
	else {
	  double Ediff=E(j)-E(k);
	  factor=(exp(-beta*E(k))-expbetaEj)*cos(Ediff*time)/(beta*Ediff);
	}
	double corr_exact=A(j,k)*A(k,j)*factor;
	corr(i)+=corr_exact;
	if (j%2==0) {
	  corr_S(i)+=corr_exact;
	}
	else {
	  corr_A(i)+=corr_exact;
	}
      }
    }
  }
  corr=(1./Z)*corr;
  corr_S=(1./Z_S)*corr_S;
  corr_A=(1./Z_A)*corr_A;
  return;
}

void corrfn0_real(vector& corr,vector& corr_S,vector& corr_A,double Z,double Z_S,double Z_A,vector E,matrix A,double beta,int size,int ntimesteps,double starttime,double dt)
  // exact zero time correlation function for
  // regular correlation function, and with symmetry
  // A must be in the same basis as D
  // < A A(t) >
{
  for (int i=0;i<ntimesteps;i++) {
    corr(i)=0.;
    corr_S(i)=0.;
    corr_A(i)=0.;
    double time=starttime+(double)i*dt;
    for (int j=0;j<size;j++) {
      double expbetaEj=exp(-beta*E(j));
      for (int k=0;k<size;k++) {
	double corr_exact;
	if (j==k) {
	  corr_exact=expbetaEj*A(j,k)*A(k,j);
	}
	else {
	  double Ediff=E(j)-E(k);
	  corr_exact=expbetaEj*A(j,k)*A(k,j)*cos(Ediff*time);
	}
	corr(i)+=corr_exact;
	if (j%2==0) {
	  corr_S(i)+=corr_exact;
	}
	else {
	  corr_A(i)+=corr_exact;
	}
      }
    }
  }
  corr=(1./Z)*corr;
  corr_S=(1./Z_S)*corr_S;
  corr_A=(1./Z_A)*corr_A;
  return;
}

void corrfn0_imag(vector& corr,vector& corr_S,vector& corr_A,double Z,double Z_S,double Z_A,vector E,matrix A,double beta,int size,int ntimesteps,double starttime,double dt)
  // exact zero time correlation function for
  // regular correlation function, and with symmetry
  // A must be in the same basis as D
  // < A A(t) >
{
  for (int i=0;i<ntimesteps;i++) {
    corr(i)=0.;
    corr_S(i)=0.;
    corr_A(i)=0.;
    double time=starttime+(double)i*dt;
    for (int j=0;j<size;j++) {
      double expbetaEj=exp(-beta*E(j));
      for (int k=0;k<size;k++) {
	if (j!=k) {
	  double Ediff=E(j)-E(k);
	  double corr_exact=expbetaEj*A(j,k)*A(k,j)*sin(Ediff*time);
	  corr(i)+=corr_exact;
	  if (j%2==0) {
	    corr_S(i)+=corr_exact;
	  }
	  else {
	    corr_A(i)+=corr_exact;
	  }
	}
      }
    }
  }
  corr=(1./Z)*corr;
  corr_S=(1./Z_S)*corr_S;
  corr_A=(1./Z_A)*corr_A;
  return;
}

vector forwardFT(vector w,int nw,vector ct,vector t,int nt,double factor)
// forward FT for a real symmetric function
{
  vector cw(nw);
  for (int iw=0;iw<nw;iw++) {
    cw(iw)=0.;
    for (int it=0;it<nt;it++) {
      if (it==0) {
	cw(iw)+=.5*ct(it);
      }
      else {
	double phase=w(iw)*t(it);
	double cosphase=cos(phase);
	cw(iw)+=cosphase*ct(it);
      }
    }
    cw(iw)*=factor;
  }
  return cw;
}

vector fwFT(vector w,int nw,vector ctR,vector ctI,vector t,int nt,double factor)
// forward FT for a complex function with a symmetric real part and an antisymmetric imaginary part
// frequency ranges from -wmax to +wmax
{
  vector cw(nw);
  for (int iw=0;iw<nw;iw++) {
    cw(iw)=0.;
    for (int it=0;it<nt;it++) {
      if (it==0) {
	cw(iw)+=.5*ctR(it);
      }
      else {
	double phase=w(iw)*t(it);
	double cosphase=cos(phase);
	double sinphase=sin(phase);
	cw(iw)+=cosphase*ctR(it)-sinphase*ctI(it);
      }
    }
    cw(iw)*=factor;
  }
  return cw;
}

vector RbackwardFT(vector t,int nt,vector cw,vector w,int nw,double factor)
// real part of backward(inverse) FT for a real function
// time (t) starts at 0
// frequency (w) is from -wmax to +wmax
{
  vector ct(nt);
  // normfactor=factor/(2 M_PI)
  double normfactor=factor*.5/M_PI;
  for (int it=0;it<nt;it++) {
    ct(it)=0.;
    for (int iw=0;iw<nw;iw++) {
      // really cosphase=cos(-wt)
      double phase=w(iw)*t(it);
      double cosphase=cos(phase);
      ct(it)+=cosphase*cw(iw);
    }
    ct(it)*=normfactor;
  }
  return ct;
}

vector IbackwardFT(vector t,int nt,vector cw,vector w,int nw,double factor)
// imaginary part of backward(inverse) FT for a real function
// time (t) starts at 0
// frequency (w) is from -wmax to +wmax
{
  vector ct(nt);
  // normfactor=factor/(2 M_PI)
  double normfactor=factor*.5/M_PI;
  for (int it=0;it<nt;it++) {
    ct(it)=0.;
    for (int iw=0;iw<nw;iw++) {
      // really sinphase=sin(-wt)
      double phase=w(iw)*t(it);
      double sinphase=-sin(phase);
      ct(it)+=sinphase*cw(iw);
    }
    ct(it)*=normfactor;
  }
  return ct;
}

vector fcw(vector w,int nw,vector cw,double beta)
{
  vector fcw(2*nw-1);
  for (int iw=0;iw<nw;iw++) {
    if (iw==0) fcw(nw-1)=cw(iw);
    else {
      double partarg=beta*w(iw);
      fcw(nw-1-iw)=-partarg/(1.-exp(partarg))*cw(iw);
      fcw(nw-1+iw)=partarg/(1.-exp(-partarg))*cw(iw);
    }
  }
  return fcw;
}

void FTcorrfn0(vector& FTcorr,vector& FTcorr_S,vector& FTcorr_A,double Z,double Z_S,double Z_A,vector E,matrix A,double beta,int size,int nw,vector w,double alpha)
{
  // exact FT of time correlation function
  // where the delta function is approximated by gaussian function
  for (int i=0;i<nw;i++) {
    FTcorr(i)=0.;
    FTcorr_S(i)=0.;
    FTcorr_A(i)=0.;
    double w2=w(i)*w(i);
    double expalphaw=exp(-alpha*w2);
    for (int j=0;j<size;j++) {
      double expbetaEj=exp(-beta*(E(j)-E(0)));
      for (int k=0;k<size;k++) {
	double FTcorr_exact;
	if (j==k) {
	  FTcorr_exact=expbetaEj*A(j,k)*A(k,j)*expalphaw;
	}
	else{
	  double Ediff=E(j)-E(k);
	  FTcorr_exact=expbetaEj*A(j,k)*A(k,j)*gaussianfn(w(i),-Ediff,alpha);
	}
	FTcorr(i)+=FTcorr_exact;
	if (j%2==0) {
	  FTcorr_S(i)+=FTcorr_exact;
	}
	else {
	  FTcorr_A(i)+=FTcorr_exact;
	}
      }
    }
  }
  double factor=sqrt(M_PI/alpha);
  FTcorr=(factor/Z)*FTcorr;
  FTcorr_S=(factor/Z)*FTcorr_S;
  FTcorr_A=(factor/Z)*FTcorr_A;
  return;
}

double gaussianfn(double x,double mu,double alpha)
{
  // returns unnormalized guassian function g(x) = exp (-alpha *(x-mu)^2 )
  // alpha >= 0
  double gaussian=exp(-alpha*pow(x-mu,2.));
  return gaussian;
}

/*
void divdynCMD(int nxcdyn,int kpindex,int ntimesteps,int accgrid,int acctrajtime,int nhist,double pcdyndiv,double dt,double threshold,double m,double corr_factor,double Z,vector xcdyn,vector hist,Interp2d rho,Interp2d f,double& Zdyndiv,vector& corr_xc,vector& corr_pc,vector& corr_xc2,vector& xct,vector& pct,vector& rhoct)
*/

void divdynCMD(int nxcdyn,int kpindex,int ntimesteps,int accgrid,int acctrajtime,double pcdyndiv,double dt,double threshold,double m,double corr_factor,double Z,vector xcdyn,Interp2d rho,Interp2d f,double& Zdyndiv,vector& corr_xc,vector& corr_pc,vector& corr_xc2)
{
  // excutes option 'divdyn' for CMD

  //stringstream trajdiv;
  //trajdiv<<"trajdiv"<<char_kpindex<<ends;
  //ofstream trajdivout(trajdiv.str().c_str());

  for (int k=0;k<(nxcdyn);k++) {
    double xc0=xcdyn(k);
    double pc0=pcdyndiv;
    double xcnew=xcdyn(k);
    double pcnew=pcdyndiv;
    double weight=rho.interp2d(xc0,pc0);

    if (kpindex==0) {
      Zdyndiv+=weight;
    }
    else {
      Zdyndiv+=2.*weight;
    }

    vector sumxcnew(ntimesteps);
    vector sumpcnew(ntimesteps);
    vector sumxc2new(ntimesteps);

    for (int i=0;i<ntimesteps;i++) {

      //if (((k+1)%accgrid==0)&&(kpindex%accgrid==0)&&(i%acctrajtime==0)) {
      //trajdivout<<(double)i*dt<<" "<<xcnew<<" "<<pcnew<<" "<<(weight_S*xcnew_S+weight_A*xcnew_A)/weight<<" "<<(weight_S*pcnew_S+weight_A*pcnew_A)/weight<<" "<<xcnew_S<<" "<<pcnew_S<<" "<<xcnew_A<<" "<<pcnew_A<<endl;
      //}

      if (fabs(weight)>threshold) {
	if (kpindex==0) {
	  sumxcnew(i)+=.5*xcnew;
	  sumpcnew(i)+=.5*pcnew;
	  sumxc2new(i)+=.5*xcnew*xcnew;

	  //corr_xc(i)+=weight*xc0*xcnew;
	  //corr_pc(i)+=weight*pc0*pcnew;
	  //corr_xc2(i)+=weight*pow(xc0*xcnew,2.);
	}
	else {
	  sumxcnew(i)+=xcnew;
	  sumpcnew(i)+=pcnew;
	  sumxc2new(i)+=xcnew*xcnew;

	  //corr_xc(i)+=2.*weight*xc0*xcnew;
	  //corr_pc(i)+=2.*weight*pc0*pcnew;
	  //corr_xc2(i)+=2.*weight*pow(xc0*xcnew,2.);
	}
	rungekutta(m,dt,f,xcnew,pcnew);
      }

      /*
      if (fabs(weight_S)>threshold) {
	if (kpindex==0) {
	  sumxcnew_S(i)+=.5*xcnew_S;
	  sumpcnew_S(i)+=.5*pcnew_S;
	  sumxc2new_S(i)+=.5*xcnew*xcnew_S;

	  //corr_xc_S(i)+=weight_S*xc0*xcnew_S;
	  //corr_pc_S(i)+=weight_S*pc0*pcnew_S;
	  //corr_xc2_S(i)+=weight_S*pow(xc0*xcnew_S,2.);
	}
	else {
	  sumxcnew_S(i)+=xcnew_S;
	  sumpcnew_S(i)+=pcnew_S;
	  sumxc2new_S(i)+=xcnew_S*xcnew_S;

	  //corr_xc_S(i)+=2.*weight_S*xc0*xcnew_S;
	  //corr_pc_S(i)+=2.*weight_S*pc0*pcnew_S;
	  //corr_xc2_S(i)+=2.*weight_S*pow(xc0*xcnew_S,2.);
	}
	rungekutta(m,dt,f_S,xcnew_S,pcnew_S);
      }
      if (fabs(weight_A)>threshold) {
	if (kpindex==0) {
	  sumxcnew_A(i)+=.5*xcnew_A;
	  sumpcnew_A(i)+=.5*pcnew_A;
	  sumxc2new_A(i)+=.5*xcnew_A*xcnew_A;

	  //corr_xc_A(i)+=weight_A*xc0*xcnew_A;
	  //corr_pc_A(i)+=weight_A*pc0*pcnew_A;
	  //corr_xc2_A(i)+=weight_A*pow(xc0*xcnew_A,2.);
	}
	else {
	  sumxcnew_A(i)+=xcnew_A;
	  sumpcnew_A(i)+=pcnew_A;
	  sumxc2new_A(i)+=xcnew_A*xcnew_A;

	  //corr_xc_A(i)+=2.*weight_A*xc0*xcnew_A;
	  //corr_pc_A(i)+=2.*weight_A*pc0*pcnew_A;
	  //corr_xc2_A(i)+=2.*weight_A*pow(xc0*xcnew_A,2.);
	}
	rungekutta(m,dt,f_A,xcnew_A,pcnew_A);
      }
      */

    }

    /*
    // temp
    xct(k)=xcnew;
    pct(k)=pcnew;
    rhoct(k)=weight;
    // end temp
    */

    double rhoxc0=2.*weight*xc0;
    double rhopc0=2.*weight*pc0;
    double rhoxc20=2.*weight*xc0*xc0;

    corr_xc=corr_xc+rhoxc0*sumxcnew;
    corr_pc=corr_pc+rhopc0*sumpcnew;
    corr_xc2=corr_xc2+rhoxc20*sumxc2new;

  }

  corr_xc=(corr_factor/Z)*corr_xc;
  corr_pc=(corr_factor/Z)*corr_pc;
  corr_xc2=(corr_factor/Z)*corr_xc2;

  Zdyndiv*=corr_factor;

  return;
}

void fulldynCMD(int nxcdyn,int npcdyn,int ntimesteps,int accgrid,int acctrajtime,double dt,double threshold,double m,double corr_factor,double Z,vector xcdyn,vector pcdyn,Interp2d rho,Interp2d f,double& Zdyn,vector& corr_xc,vector& corr_pc,vector& corr_xc2)
{
  // excutes option 'fulldyn' for CMD

  //ofstream trajout("traj");

  for (int k=0;k<(nxcdyn);k++) {
    for (int kp=0;kp<(npcdyn);kp++) {
      double xc0=xcdyn(k);
      double pc0=pcdyn(kp);
      double xcnew=xcdyn(k);
      double pcnew=pcdyn(kp);
      double weight=rho.interp2d(xc0,pc0);

      //double xcnew_S=xcdyn(k);
      //double pcnew_S=pcdyn(kp);
      //double weight_S=rho_S.interp2d(xc0,pc0);

      //double xcnew_A=xcdyn(k);
      //double pcnew_A=pcdyn(kp);
      //double weight_A=rho_A.interp2d(xc0,pc0);

      //double xclnnew=xcdyn(k);
      //double pclnnew=pcdyn(kp);

      //double xclnnew_S=xcdyn(k);
      //double pclnnew_S=pcdyn(kp);

      //double xclnnew_A=xcdyn(k);
      //double pclnnew_A=pcdyn(kp);

      Zdyn+=weight;
      //Zdyn_S+=weight_S;
      //Zdyn_A+=weight_A;

      vector sumxcnew(ntimesteps);
      vector sumpcnew(ntimesteps);
      vector sumxc2new(ntimesteps);

      for (int i=0;i<ntimesteps;i++) {

	//if (((k+1)%accgrid==0)&&(kpindex%accgrid==0)&&(i%acctrajtime==0)) {
	//trajout<<(double)i*dt<<" "<<xcnew<<" "<<pcnew<<" "<<(weight_S*xcnew_S+weight_A*xcnew_A)/weight<<" "<<(weight_S*pcnew_S+weight_A*pcnew_A)/weight<<" "<<xcnew_S<<" "<<pcnew_S<<" "<<xcnew_A<<" "<<pcnew_A<<" "<<xclnnew<<" "<<pclnnew<<" "<<xclnnew_S<<" "<<pclnnew_S<<" "<<xclnnew_A<<" "<<pclnnew_A<<endl;
	//}

	if (fabs(weight)>threshold) {

	  sumxcnew(i)+=xcnew;
	  sumpcnew(i)+=pcnew;
	  sumxc2new(i)+=xcnew*xcnew;

	  //corr_xc(i)+=weight*xc0*xcnew;
	  //corr_xcln(i)+=weight*xc0*xclnnew;

	  //corr_pc(i)+=weight*pc0*pcnew;
	  //corr_pcln(i)+=weight*pc0*pclnnew;

	  //corr_xc2(i)+=weight*pow(xc0*xcnew,2.);
	  //corr_xc2ln(i)+=weight*pow(xc0*xclnnew,2.);

	  // Euler
	  //double force=f.interp2d(xcnew,pcnew);
	  //double forceln=fln.interp2d(xclnnew,pclnnew);
	  //euler(m,dt,force,xcnew,pcnew);
	  //euler(m,dt,forceln,xclnnew,pclnnew);
	  
	  // Runga Kutta
	  rungekutta(m,dt,f,xcnew,pcnew);
	  //rungekutta(m,dt,fln,xclnnew,pclnnew);
	}

	/*
	if (fabs(weight_S)>threshold) {
	  corr_xc_S(i)+=weight_S*xc0*xcnew_S;
	  corr_xcln_S(i)+=weight_S*xc0*xclnnew_S;

	  corr_pc_S(i)+=weight_S*pc0*pcnew_S;
	  corr_pcln_S(i)+=weight_S*pc0*pclnnew_S;

	  corr_xc2_S(i)+=weight_S*pow(xc0*xcnew_S,2.);
	  corr_xc2ln_S(i)+=weight_S*pow(xc0*xclnnew_S,2.);
  
	  // Euler
	  //double force_S=f_S.interp2d(xcnew_S,pcnew_S);
	  //double forceln_S=fln_S.interp2d(xclnnew_S,pclnnew_S);
	  //euler(m,dt,force_S,xcnew_S,pcnew_S);
	  //euler(m,dt,forceln_S,xclnnew_S,pclnnew_S);

	  // Runga Kutta
	  rungekutta(m,dt,f_S,xcnew_S,pcnew_S);
	  rungekutta(m,dt,fln_S,xclnnew_S,pclnnew_S);
	}
	if (fabs(weight_A)>threshold) {
	  corr_xc_A(i)+=weight_A*xc0*xcnew_A;
	  corr_xcln_A(i)+=weight_A*xc0*xclnnew_A;

	  corr_pc_A(i)+=weight_A*pc0*pcnew_A;
	  corr_pcln_A(i)+=weight_A*pc0*pclnnew_A;

	  corr_xc2_A(i)+=weight_A*pow(xc0*xcnew_A,2.);
	  corr_xc2ln_A(i)+=weight_A*pow(xc0*xclnnew_A,2.);

	  // Euler
	  //double force_A=f_A.interp2d(xcnew_A,pcnew_A);
	  //double forceln_A=fln_A.interp2d(xclnnew_A,pclnnew_A);
	  //euler(m,dt,force_A,xcnew_A,pcnew_A);
	  //euler(m,dt,forceln_A,xclnnew_A,pclnnew_A);

	  // Runga Kutta
	  rungekutta(m,dt,f_A,xcnew_A,pcnew_A);
	  rungekutta(m,dt,fln_A,xclnnew_A,pclnnew_A);
	}
	*/


      }

      double rhoxc0=weight*xc0;
      double rhopc0=weight*pc0;
      double rhoxc20=weight*xc0*xc0;

      corr_xc=corr_xc+rhoxc0*sumxcnew;
      corr_pc=corr_pc+rhopc0*sumpcnew;
      corr_xc2=corr_xc2+rhoxc20*sumxc2new;

    }
  }

  corr_xc=(corr_factor/Z)*corr_xc;
  //corr_xcln=(corr_factor/Z)*corr_xcln;
  //corr_xc_S=(corr_factor/Z_S)*corr_xc_S;
  //corr_xcln_S=(corr_factor/Z_S)*corr_xcln_S;
  //corr_xc_A=(corr_factor/Z_A)*corr_xc_A;
  //corr_xcln_A=(corr_factor/Z_A)*corr_xcln_A;

  corr_pc=(corr_factor/Z)*corr_pc;
  //corr_pcln=(corr_factor/Z)*corr_pcln;
  //corr_pc_S=(corr_factor/Z_S)*corr_pc_S;
  //corr_pcln_S=(corr_factor/Z_S)*corr_pcln_S;
  //corr_pc_A=(corr_factor/Z_A)*corr_pc_A;
  //corr_pcln_A=(corr_factor/Z_A)*corr_pcln_A;

  corr_xc2=(corr_factor/Z)*corr_xc2;
  //corr_xc2ln=(corr_factor/Z)*corr_xc2ln;
  //corr_xc2_S=(corr_factor/Z_S)*corr_xc2_S;
  //corr_xc2ln_S=(corr_factor/Z_S)*corr_xc2ln_S;
  //corr_xc2_A=(corr_factor/Z_A)*corr_xc2_A;
  //corr_xc2ln_A=(corr_factor/Z_A)*corr_xc2ln_A;

  // Z is multiplied by 1/(2*M_PI) *dxc*dpc
  Zdyn*=corr_factor;
  //Zdyn_S*=corr_factor;
  //Zdyn_A*=corr_factor;

  return;
}

