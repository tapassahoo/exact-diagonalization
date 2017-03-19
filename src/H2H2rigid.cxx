#include "cmdstuff.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sstream>

using namespace std;

static const double hatocm=219474.63067;

double Vinit();
EXTERN void FORTRAN(vinit)();
EXTERN void FORTRAN(vh2h2pieces)(double *RR,double *rr1,double *rr2,double *tth1,double *tth2,double *pphi,double *Eint, double *potl);
EXTERNC void gauleg(double x1,double x2,double *x,double *w,int n);
EXTERNC double plgndr(int j,int m,double x);
vector thetagrid(int nsize,vector &weights);
vector phigrid(int nsize,vector &weights);
double PjNormal(int j,int m, double x);
double Pj0(int j,double x);
void fbasisT1(int jmax,int size_theta1,vector &weights_theta1,vector &grid_theta1,matrix &basisT1);
void fbasisT2(int jmax,int size_theta2,vector &weights_theta2,vector &grid_theta2,matrix &basisT2);
void fbasisP(int jmax,int size_phi,vector &weights_phi,vector &grid_phi,matrix &basisP);
void get_sizes(int jmax, int *sizes);
void testcall();
double basisPjm(int j,int m,double w,double x);
double basisfunctionP(int m,double w,double phi);

int main(int argc,char **argv) {
  int i,j,k,j1,j1p,j2,j2p,jp,n1,n2,n1p,n2p;
  time_t totalstart,totalend,callpotstart,callpotend,diagstart,diagend;
  time (&totalstart);
  char timemsg[100];
	
  //	Vinit(); // initialize coefficients for 6d potential
  vinit_(); // initialize coefficients for 6d potential

  //  testcall();
  //exit(0);

  if (argc != 2) {
    cerr<<"usage: "<<argv[0]<<" <R value> in bohrs"<<endl;
    exit(0);
  }
  //double Rpt=atof(argv[1]);
  double Rpt=atof(argv[1])*atob;

  int sizej=3;

  int jmax=2*(sizej-1);

  // ******************************
  // Set up basis for theta and phi
  // ******************************

  int size_theta1=2.*jmax+5;
  int size_theta2=2.*jmax+5;
  int size_phi=2.*(2.*jmax+7); // 

  // define grid points for theta1 theta2 and phi
  // initialize vectors of weights

  vector weights_theta1(size_theta1);
  vector weights_theta2(size_theta2);
  vector weights_phi(size_phi); 

  // set vectors of grid and weights

  vector grid_theta1=thetagrid(size_theta1,weights_theta1);
  vector grid_theta2=thetagrid(size_theta2,weights_theta2);
  vector grid_phi=phigrid(size_phi,weights_phi);

  // evaluate basis fxns on grid points
  int sizes[2];
  get_sizes(jmax,sizes);

  ofstream logout("log");

  // test orthonormality of theta and phi basis
  logout<<"  // test orthonormality of theta and phi basis"<<endl;
  for (int m1=-jmax;m1<=jmax;m1++) {
    for (j1=abs(m1);j1<=jmax;j1++){
      if (j1%2) continue; // skip iteration if j1 odd
      int m2=m1;
      for (j2=abs(m2);j2<=jmax;j2++){
	if (j2%2) continue; // skip iteration if j2 odd
	double sum=0.;
	for (i=0;i<size_theta1;i++){
	  double theta1=grid_theta1(i);
	  sum+=basisPjm(j1,m1,weights_theta1(i),cos(theta1))*basisPjm(j2,m2,weights_theta1(i),cos(theta1));
	}
	logout<<" m1 = "<<m1<<" "<<j1<<" "<<j2<<" "<<sum<<endl;
      }
    }
  }

  for (int m1=-jmax;m1<=jmax;m1++) {
    for (int m1p=-jmax;m1p<=jmax;m1p++) {
      double sum=0.;
      for (k=0;k<size_phi;k++){
	double phi=grid_phi(k);
	sum+=basisfunctionP(m1,weights_phi(k),phi)*basisfunctionP(m1p,weights_phi(k),phi);
      }
      logout<<" m1,m1p = "<<m1<<" "<<m1p<<" "<<sum<<endl;      
    }
  }
 
  // **************************************
  // Define kinetic energy info
  // **************************************

  // the rovibrational energy levels for the monomers
  vector E0j(3);

  E0j(0)=-36118.074;
  E0j(1)=-35763.701;
  E0j(2)=-34949.276;
	
  double B=(E0j(1)-E0j(0))/6.;

  B=59.2466724;
  
  // *********
  // Main loop
  // *********

  int sizej1j2m=sizes[1];


  int t1_i,t2_i,p_i,a,b,m,mp;
  double r1,r2;
  double theta1,theta2,phi;
  double signm,signmp;
  double V6d=0.;
  double norm_1d=0.;
  
  //calculate r1,r2 part of potential on the grid

 
  // use dummy theta's and phi
  theta1=0.;
  theta2=0.;
  phi=0.;
  double potl[8]; //array to store pieces of potential
  double A000;
  double A022;
  double A202;
  double A224;
  r1=1.42; //bohrs
  r2=1.42; //bohrs  
  vh2h2pieces_(&Rpt,&r1,&r2,&theta1,&theta2,&phi,&V6d,potl);
  A000=potl[0];
  A022=potl[1];
  A202=potl[2];
  A224=potl[3];

  // calculate theta1,theta2,phi part on the grid
  // use dummy r's
  r1=1.42;
  r2=1.42;
  matrix G000(size_theta1*size_theta2,size_phi);
  matrix G022(size_theta1*size_theta2,size_phi);
  matrix G202(size_theta1*size_theta2,size_phi);
  matrix G224(size_theta1*size_theta2,size_phi);
  for (i=0;i<size_theta1;i++){
    theta1=grid_theta1(i);
    for (j=0;j<size_theta2;j++){
      theta2=grid_theta2(j);
      for (k=0;k<size_phi;k++){
	phi=grid_phi(k);
	vh2h2pieces_(&Rpt,&r1,&r2,&theta1,&theta2,&phi,&V6d,potl);
	G000(i*size_theta2+j,k)=potl[4];
	G022(i*size_theta2+j,k)=potl[5];
	G202(i*size_theta2+j,k)=potl[6];
	G224(i*size_theta2+j,k)=potl[7];
      }
    }
  }

  // angular part
  int index_m1=0;
  int index_m1p=0;

  vector Dphi_G000(size_theta1*size_theta2);
  vector Dphi_G022(size_theta1*size_theta2);
  vector Dphi_G202(size_theta1*size_theta2);
  vector Dphi_G224(size_theta1*size_theta2);

  vector Dtheta_G000(size_theta1);
  vector Dtheta_G022(size_theta1);
  vector Dtheta_G202(size_theta1);
  vector Dtheta_G224(size_theta1);

  matrix G000_basis(sizej1j2m,sizej1j2m);
  matrix G022_basis(sizej1j2m,sizej1j2m);
  matrix G202_basis(sizej1j2m,sizej1j2m);
  matrix G224_basis(sizej1j2m,sizej1j2m);
  
  int index_jm12=-1;
  for (int m1=-jmax;m1<=jmax;m1++) {
    for (j1=abs(m1);j1<=jmax;j1++){
      if (j1%2) continue; // skip iteration if j1 odd
      int m2=-m1;
      for (j2=abs(m2);j2<=jmax;j2++){
	if (j2%2) continue; // skip iteration if j2 odd
	index_jm12++;
	int index_jm12p=-1;
	for (int m1p=-jmax;m1p<=jmax;m1p++) {
	  for (j1p=abs(m1p);j1p<=jmax;j1p++){
	    if (j1p%2) continue; // skip iteration if j1 odd
	    int m2p=-m1p;
	    for (j2p=abs(m2p);j2p<=jmax;j2p++){
	      if (j2p%2) continue; // skip iteration if j2 odd
	      index_jm12p++;
	      // theta1 quadrature
	      for (i=0;i<size_theta1;i++){
		theta1=grid_theta1(i);
		// theta2 quadrature
		Dtheta_G000(i)=0.;
		Dtheta_G022(i)=0.;
		Dtheta_G202(i)=0.;
		Dtheta_G224(i)=0.;
		for (j=0;j<size_theta2;j++){
		  theta2=grid_theta2(j);
		  // phi quadrature
		  Dphi_G000(i*size_theta2+j)=0.;
		  Dphi_G022(i*size_theta2+j)=0.;
		  Dphi_G202(i*size_theta2+j)=0.;
		  Dphi_G224(i*size_theta2+j)=0.;
		  for (k=0;k<size_phi;k++){
		    phi=grid_phi(k);
		    Dphi_G000(i*size_theta2+j)+=G000(i*size_theta2+j,k)*basisfunctionP(m1,weights_phi(k),phi)*basisfunctionP(m1p,weights_phi(k),phi);
		    Dphi_G022(i*size_theta2+j)+=G022(i*size_theta2+j,k)*basisfunctionP(m1,weights_phi(k),phi)*basisfunctionP(m1p,weights_phi(k),phi);
		    Dphi_G202(i*size_theta2+j)+=G202(i*size_theta2+j,k)*basisfunctionP(m1,weights_phi(k),phi)*basisfunctionP(m1p,weights_phi(k),phi);
		    Dphi_G224(i*size_theta2+j)+=G224(i*size_theta2+j,k)*basisfunctionP(m1,weights_phi(k),phi)*basisfunctionP(m1p,weights_phi(k),phi);
		  }
		  Dtheta_G000(i)+=Dphi_G000(i*size_theta2+j)*basisPjm(j2,m2,weights_theta1(j),cos(theta2))*basisPjm(j2p,m2p,weights_theta1(j),cos(theta2));
		  Dtheta_G022(i)+=Dphi_G022(i*size_theta2+j)*basisPjm(j2,m2,weights_theta1(j),cos(theta2))*basisPjm(j2p,m2p,weights_theta1(j),cos(theta2));
		  Dtheta_G202(i)+=Dphi_G202(i*size_theta2+j)*basisPjm(j2,m2,weights_theta1(j),cos(theta2))*basisPjm(j2p,m2p,weights_theta1(j),cos(theta2));
		  Dtheta_G224(i)+=Dphi_G224(i*size_theta2+j)*basisPjm(j2,m2,weights_theta1(j),cos(theta2))*basisPjm(j2p,m2p,weights_theta1(j),cos(theta2));
		}
		G000_basis(index_jm12,index_jm12p)+=Dtheta_G000(i)*basisPjm(j1,abs(m1),weights_theta1(i),cos(theta1))*basisPjm(j1p,abs(m1p),weights_theta1(i),cos(theta1));
		G022_basis(index_jm12,index_jm12p)+=Dtheta_G022(i)*basisPjm(j1,abs(m1),weights_theta1(i),cos(theta1))*basisPjm(j1p,abs(m1p),weights_theta1(i),cos(theta1));
		G202_basis(index_jm12,index_jm12p)+=Dtheta_G202(i)*basisPjm(j1,abs(m1),weights_theta1(i),cos(theta1))*basisPjm(j1p,abs(m1p),weights_theta1(i),cos(theta1));
		G224_basis(index_jm12,index_jm12p)+=Dtheta_G224(i)*basisPjm(j1,abs(m1),weights_theta1(i),cos(theta1))*basisPjm(j1p,abs(m1p),weights_theta1(i),cos(theta1));
	      }
	    }
	  }
	}
      }
    }
  }
    
  // setup H

  //cout<<size_nu<<" "<<sizej1j2m<<" "<<jmax<<endl;
  matrix H(sizej1j2m,sizej1j2m);
  matrix V(sizej1j2m,sizej1j2m);
  matrix delta(sizej1j2m,sizej1j2m);

  ofstream Vout("V");
  ofstream Hout("H");

  index_jm12=-1;
  for (int m1=-jmax;m1<=jmax;m1++) {
    for (j1=abs(m1);j1<=jmax;j1++){
      if (j1%2) continue; // skip iteration if j1 odd
      int m2=-m1;
      for (j2=abs(m2);j2<=jmax;j2++){
	if (j2%2) continue; // skip iteration if j2 odd
	index_jm12++;
	H(index_jm12,index_jm12)=B*((double)(j1*(j1+1))+(double)(j2*(j2+1)));
	int index_jm12p=-1;
	for (int m1p=-jmax;m1p<=jmax;m1p++) {
	  for (j1p=abs(m1p);j1p<=jmax;j1p++){
	    if (j1p%2) continue; // skip iteration if j1 odd
	    int m2p=-m1p;
	    for (j2p=abs(m2p);j2p<=jmax;j2p++){
	      if (j2p%2) continue; // skip iteration if j2 odd
	      index_jm12p++;		       
	      
	      V(index_jm12,index_jm12p)=
		A000*G000_basis(index_jm12,index_jm12p)
		+A022*G022_basis(index_jm12,index_jm12p)
		+A202*G202_basis(index_jm12,index_jm12p)
		+A224*G224_basis(index_jm12,index_jm12p);	      	      	   

	      // diagonal part
	      if (j1==j1p && j2==j2p && m1==(m1p))
		delta(index_jm12,index_jm12)=.5;

	      // P12
	      if (j1==j2p && j2==j1p && m1==m2p && m2==m1p)
		delta(index_jm12,index_jm12p)+=.5;
	    }
	  }
	}
      }
    }
  }

  // Diagonalize

  time (&diagstart);
  
  H=H+V;

  //  ofstream Hout("H");
  for (i=0;i<sizej1j2m;i++){
    for (j=0;j<sizej1j2m;j++)
      Hout<<H(i,j)<<" ";
     Hout<<endl;
  }
  
  matrix Hsym=delta*(H*delta);

  vector evdelta=diag(delta);
  for (i=0;i<sizej1j2m;i++){
    //cout<<evdelta(i)<<endl;
  }
  
  vector evsym=diag(Hsym);
  vector ev=diag(H);
  time (&diagend);
  double dif2 = difftime (diagend,diagstart);
  sprintf (timemsg, "Elapsed time for diagonalization is %.2lf seconds.", dif2);
  logout<<timemsg<<endl;

  ofstream evalues("ev");
  evalues.precision(3);
  for (i=0;i<sizej1j2m;i++){
    evalues<<ev(i)<<"   ";
    for (j=0;j<sizej1j2m;j++)
      evalues<<H(j,i)<<" ";
     evalues<<endl;
  }
  ofstream evaluessym("evsym");
  evaluessym.precision(3);
  for (i=0;i<sizej1j2m;i++){
    evaluessym<<evsym(i)<<"   ";
    for (j=0;j<sizej1j2m;j++)
      evaluessym<<Hsym(j,i)<<" ";
     evaluessym<<endl;
  }

	matrix avgV=	transpose(H)*V*H;
  logout<<"B= "<<B<<" cm-1"<<endl;
  logout<<"r1 = "<<r1<<" bohrs"<<endl;
  logout<<"r2 = "<<r2<<" bohrs"<<endl;
  logout<<"R = "<<Rpt<<" bohrs"<<endl;
  logout<<"E_0 = "<<ev(0)<<" cm-1"<<endl;
  logout<<"E_0 (sym) = "<<evsym(0)<<" cm-1"<<endl;

  cout<<Rpt/atob<<" "<<ev(0)/0.695<<" "<<evsym(0)/0.695<<"  "<<avgV(0,0)/0.695<<"     "<<H(0,0)<<endl;

  time (&totalend);
  double dif3 = difftime (totalend,totalstart);
  sprintf (timemsg, "Total elapsed time is %.2lf seconds.\n", dif3);
  logout<<timemsg<<endl;
}

double Vinit() {
  FORTRAN(vinit)();
  return 0;
}

vector thetagrid(int nsize,vector &weights)
{
  int i;
  vector grid(nsize);
  double *x=new double[nsize];
  double *w=new double[nsize];
  double x1=-1.;
  double x2=1.;
  gauleg(x1,x2,x,w,nsize);
  for (i=0;i<nsize;i++) {
    grid(i)=acos(x[i]);
    weights(i)=w[i]; // in this case weights_theta=weights since the argument uses a reference operator
  }
  return grid;
}

vector phigrid(int nsize,vector &weights)
{
  int i;
  vector grid(nsize);

  double phimin=0.*M_PI;
  double phimax=2.*M_PI;
  double dphi=2.*M_PI/(double)nsize;
  for (i=0;i<nsize;i++) {
        grid(i)=phimin+((double)i)*dphi;
        weights(i)=dphi;
  }
  return grid;

}
// real spherical harmonics
//
// Ylm= sqrt(2) sqrt[ (2l+1)/4Pi (l-|m|)!/(l+|m|)! Pl^|m|(cos theta) sin(|m| phi) ] m<0
// Ylm=  sqrt[ (2l+1)/2Pi (l-|m|)!/(l+|m|)! Pl^|m|(cos theta) sin(|m| phi) ] m<0
// Ylm=  sqrt[ (l+.5)/Pi (l-|m|)!/(l+|m|)! Pl^|m|(cos theta) sin(|m| phi) ] m<0
// Ylm=  (1/sqrt(Pi)) sqrt[ (l+.5) (l-|m|)!/(l+|m|)! Pl^|m|(cos theta) sin(|m| phi) ] m<0
//
//  = sqrt[ (2l+1)/4Pi] Pl^0(cos theta)
//  = (1/sqrt(2Pi)) sqrt[ (l+.5)] Pl^0(cos theta)
//
//  = sqrt(2) sqrt[ (2l+1)/4Pi (l-m)!/(l+m)! Pl^m(cos theta) cos(m phi) ] m>0
//  = (1/sqrt(Pi))  sqrt[ (l+.5) (l-m)!/(l+m)! Pl^m(cos theta) cos(m phi) ] m>0
//
// (12) P(l1,|m1|) T(m1) P(l2,m2) -> P(l2,|m2|) T(m2) P(l1,m1)



double PjNormal(int j,int m, double x)
{
  if (j < m) cerr<<"j < m"<<endl;
  double jv=(double)j;
  double mv=fabs((double)m);
  double PjN= pow(-1.,mv)*sqrt((jv+.5)*exp(lgamma(jv-mv+1.)-lgamma(jv+mv+1.)))*plgndr(j,abs(m),x); // for the normalization constant sqrt(...) refer to 6.55 in Quantum Chemistry. lgamma is the natural logarithim of the gamma function: gamma(n) = (n-1)!
  if (m<0)
    PjN=pow(-1.,mv)*PjN; // refer to pg. 9 Angular Momentum, Richard N. Zare 
  return PjN;

}

double Pj0(int j,double x)
{
  return sqrt((double)j+.5)*plgndr(j,0,x);
}

void fbasisT1(int jmax,int size_theta1,vector &weights_theta1,vector &grid_theta1,matrix &basisT1)
{
  int j1,m,t1_i;
  double phase;

  int indexmj1=0;
  for (m=-jmax;m<=jmax;m++) {
    if (m>0) phase=pow((-1.),(m));
    else phase=1.;
    for (j1=abs(m);j1<=jmax;j1++) { // the range of j is defined by the current m. eg. for m=1, j=1,...,jmax but j != 0
							
      if (j1%2) continue; // skip iteration if j1 odd
      for(t1_i=0;t1_i<size_theta1;t1_i++){
	basisT1(indexmj1,t1_i)=sqrt(weights_theta1(t1_i))*phase*sqrt(1./(2.*M_PI))*PjNormal(j1,m,cos(grid_theta1(t1_i)));
      }
      indexmj1++;
    }
  }

  return;
}

void fbasisT2(int jmax,int size_theta2,vector &weights_theta2,vector &grid_theta2,matrix &basisT2)
{
  int j2,m,t2_i;
  double phase;

  int indexmj2=0;
  for (m=-jmax;m<=jmax;m++) {
    if (m>0) phase=pow((-1.),(m));
    else phase=1.;
    for (j2=abs(m);j2<=jmax;j2++) { // the range of j is defined by the current m. eg. for m=1, j=1,...,jmax but j != 0
							
      if (j2%2) continue; // skip iteration if j2 odd
      for(t2_i=0;t2_i<size_theta2;t2_i++){
	basisT2(indexmj2,t2_i)=sqrt(weights_theta2(t2_i))*phase*sqrt(1./(2.*M_PI))*PjNormal(j2,m,cos(grid_theta2(t2_i)));
      }
      // cout<<"m: "<<m<<" j2: "<<j2<<" indexmj2: "<<indexmj2<<endl;
      indexmj2++;
    }
  }

  return;
}

double basisPjm(int j,int m,double w,double x)
{
  double phase;
  double value;

  if (m>0) phase=pow((-1.),(m)); else phase=1.;							
  value=sqrt(w)*phase*PjNormal(j,m,x);
      //      value=sqrt(w)*phase*sqrt(1./(2.*M_PI))*PjNormal(j,m,x);
    
  return value;
}

void fbasisP(int jmax,int size_phi,vector &weights_phi,vector &grid_phi,matrix &basisP)
{
  int m,p_i;
  int indexm=0;
  for (m=-jmax;m<=jmax;m++) {
    for (p_i=0;p_i<size_phi;p_i++) {
      if (m>0)
	basisP(indexm,p_i)=sqrt(1./(2.*M_PI))*sqrt(weights_phi(p_i))*cos(m*grid_phi(p_i));
      if (m==0)
	basisP(indexm,p_i)=1.;
      if (m<0)
	basisP(indexm,p_i)=sqrt(1./(2.*M_PI))*sqrt(weights_phi(p_i))*sin(abs(m)*grid_phi(p_i));
    }
    indexm++;
  }
}
double basisfunctionP(int m,double w,double phi)
{
  double value;
  if (m>0)
    value=sqrt(1./(M_PI))*sqrt(w)*cos((double)m*phi);
  if (m==0)
    value=sqrt(1./(2.*M_PI))*sqrt(w);
  if (m<0)
    value=sqrt(1./(M_PI))*sqrt(w)*sin((double)abs(m)*phi);
  return value;
}

void get_sizes(int jmax, int *sizes)
{
  int j1,j2,m;
  int indexmj=0;
  int indexmj1j2=0;
  int indexmj2_out=0;
  for (m=-jmax;m<=jmax;m++) {
    for (j1=abs(m);j1<=jmax;j1++) {
							
      if (j1%2) continue; // skip iteration if j odd

      int indexmj2_in=0;
      for (j2=abs(m);j2<=jmax;j2++) {

	if (j2%2) continue;
	// cout<<"m: "<<m<<" j1: "<<j1<<" j2: "<<j2<<" indexmj2_out: "<<indexmj2_out<<" indexmj2_in: "<<indexmj2_in<<" indexmj2_out+indexmj2_in: "<<indexmj2_out+indexmj2_in<<endl;
	indexmj2_in++;
	if (j1==jmax && j2==jmax) indexmj2_out+=indexmj2_in;
	indexmj1j2++;
      }
      indexmj++;
    }
  }
  sizes[0]=indexmj+1; // sizemj
  sizes[1]=indexmj1j2; // sizej1j2m

  return;
}
void testcall()
{
  double Rpt=6.;
  double r1=1.4;
  double r2=1.4;
  double theta1=0.3;
  double theta2=0.7;
  double phi=2.;
  double V6d;
  double potl[8];
  vh2h2pieces_(&Rpt,&r1,&r2,&theta1,&theta2,&phi,&V6d,potl);
  cout<<"A000 value= "<<potl[0]<<endl;
  cout<<"A022 value= "<<potl[1]<<endl;
  cout<<"A202 value= "<<potl[2]<<endl;
  cout<<"A224 value= "<<potl[3]<<endl;
  cout<<"G000 value= "<<potl[4]<<endl;
  cout<<"G022 value= "<<potl[5]<<endl;
  cout<<"G202 value= "<<potl[6]<<endl;
  cout<<"G224 value= "<<potl[7]<<endl;
  return;
}
