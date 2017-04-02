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
matrix fbasisP(int jmax,int size_phi,vector &weights_phi,vector &grid_phi);
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

  if (argc != 3) {
    cerr<<"usage: "<<argv[0]<<" <R value> in A <size nu_total> 1, 2, or 4"<<endl;
    exit(0);
  }
  double Rpt=atof(argv[1])/BohrToA;
	
  // *******************************
  // Read wavefunctions for each v,j
  // *******************************

  // for averaging over r. Store them in 4 column matrices as (r, wavefxn
  // amplitude for each j). The small r grid must be the same for all and is stored in
  // first column.

  int sizej=3;

  int jmax=2*(sizej-1);

  
  int size_w=848;

  matrix wave0(size_w,sizej+1);
	
  for (j=0;j<sizej;j++) {
    stringstream w0in;
    w0in << "H2_v0j" << j*2 << ".dat";
    ifstream wave0_in(w0in.str().c_str());
    for (i=0;i<size_w;i++)
      wave0_in>>wave0(i,0)>>wave0(i,j+1);
    wave0_in.close();
  }
	
  matrix wave1(size_w,sizej+1);
	
  for (j=0;j<sizej;j++) {
    stringstream w1in;
    w1in << "H2_v1j" << j*2 << ".dat";
    ifstream wave1_in(w1in.str().c_str());	
    for (i=0;i<size_w;i++)
      wave1_in>>wave1(i,0)>>wave1(i,j+1);
    wave1_in.close();
  }

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
  int sizes[1];
  get_sizes(jmax,sizes);



  matrix basisP=fbasisP( jmax,size_phi,weights_phi,grid_phi);


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
  vector E1j(3);
  vector Enuj(2*sizej);

  E0j(0)=-36118.074;
  E0j(1)=-35763.701;
  E0j(2)=-34949.276;
	
  E1j(0)=-31956.927;
  E1j(1)=-31620.254;
  E1j(2)=-30846.711;


  
  int nu=0;
  for (j=0;j<sizej;j++)
    Enuj(nu*sizej+j)=E0j(j);
  nu=1;
  for (j=0;j<sizej;j++)
    Enuj(nu*sizej+j)=E1j(j);


  
  double T; // will hold the kinetic part of the Hamiltonian
  
  // *********
  // Main loop
  // *********

  int sizej1j2m=sizes[0]*sizes[0];

  // nu_total=0, nutotal=1 nutotal <=1
  int size_nu=atoi(argv[2]);

  int nulist1[size_nu];
  int nulist2[size_nu];
  if (size_nu==1) {
    nulist1[0]=0;
    nulist2[0]=0;
  }
  if (size_nu==2) {
    nulist1[0]=0;
    nulist2[0]=1;
    nulist1[1]=1;
    nulist2[1]=0;
  }
  if (size_nu==4) {
   nulist1[0]=0;
   nulist2[0]=0; 
   nulist1[1]=0;
   nulist2[1]=1;
   nulist1[2]=1;
   nulist2[2]=0;
   nulist1[3]=1;
   nulist2[3]=1;
  }
  

  int t1_i,t2_i,p_i,a,b,m,mp;
  double r1,r2;
  double theta1,theta2,phi;
  double signm,signmp;
  double del_r=(wave0(1,0)-wave0(0,0)); // grid spacing of r
  double V6d=0.;
  double V1dBra,V1dKet,V1d;
  double norm_1d=0.;


  // Gram-Schmidt
  // test orthogonality
  logout<<"  // test orthogonality of r1,r2 bases"<<endl;
  for (j=0;j<sizej;j++) {
    double s01=0;
    for (int n=0;n<size_w;n++) {
      s01+=wave1(n,j+1)*wave0(n,j+1)*del_r;
    }
    logout<<"overlap <0|1> "<<j<<" "<<s01<<endl;
    // gram-schmidt ortho
    for (int n=0;n<size_w;n++) 
      wave1(n,j+1)-=s01*wave0(n,j+1);
  }
	
  // Normalization factors of wavefunctions
  
  double nw0;
  double nw1;
  
  for (int j=0;j<sizej;j++) {
    nw0=0.;
    nw1=0.;
    for (int n=0;n<size_w;n++) {
      nw0+=wave0(n,j+1)*wave0(n,j+1)*del_r;
      nw1+=wave1(n,j+1)*wave1(n,j+1)*del_r;
    }
    nw0=sqrt(nw0);
    nw1=sqrt(nw1);
    // renormalize
    for (int n=0;n<size_w;n++) 
      wave1(n,j+1)/=nw1;
  }

  //test normalization
  for (int j=0;j<sizej;j++) {
    nw0=0.;
    nw1=0.;
    for (int n=0;n<size_w;n++) {
      nw0+=wave0(n,j+1)*wave0(n,j+1)*del_r;
      nw1+=wave1(n,j+1)*wave1(n,j+1)*del_r;
    }
    nw0=sqrt(nw0);
    nw1=sqrt(nw1);
    logout<<"norm for nu=0 and j = "<<j*2<<" is "<< nw0<<endl;
    logout<<"norm for nu=1 and j = "<<j*2<<" is "<< nw1<<endl;
  }
  // re-test orthogonality after  Gram-Schmidt
  logout<<"  // re-test orthogonality after  Gram-Schmidt"<<endl;
  for (j=0;j<sizej;j++) {
    double s01=0;
    for (int n=0;n<size_w;n++) {
      s01+=wave1(n,j+1)*wave0(n,j+1)*del_r;
    }
    logout<<"new overlap <0|1> "<<j<<" "<<s01<<endl;
    // another gram-schmidt ortho
    for (int n=0;n<size_w;n++)
      wave1(n,j+1)-=s01*wave0(n,j+1);
  }
  //test normalization
  for (int j=0;j<sizej;j++) {
    nw0=0.;
    nw1=0.;
    for (int n=0;n<size_w;n++) {
      nw0+=wave0(n,j+1)*wave0(n,j+1)*del_r;
      nw1+=wave1(n,j+1)*wave1(n,j+1)*del_r;
    }
    nw0=sqrt(nw0);
    nw1=sqrt(nw1);
    // renormalize
    logout<<"norm for nu=0 and j = "<<j*2<<" is "<< nw0<<endl;
    logout<<"norm for nu=1 and j = "<<j*2<<" is "<< nw1<<endl;
    for (int n=0;n<size_w;n++) 
      wave1(n,j+1)/=nw1;
  }
  // re-re-test orthogonality
  logout<<"  // re-re-test orthogonality after another  Gram-Schmidt"<<endl;
  for (j=0;j<sizej;j++) {
    double s01=0;
    for (int n=0;n<size_w;n++) {
      s01+=wave1(n,j+1)*wave0(n,j+1)*del_r;
    }
    logout<<"new overlap <0|1> "<<j*2<<" "<<s01<<endl;
  }

  
  matrix psioverlaps(2*sizej,2*sizej);
  logout<<" psi overlaps "<<endl;
  for (j=0;j<sizej;j++) {
    for (jp=0;jp<sizej;jp++) {
      for (int n=0;n<size_w;n++) {
	int nu=0;
	int nup=0;
	psioverlaps(nu*sizej+j,nup*sizej+jp)+=wave0(n,j+1)*wave0(n,jp+1)*del_r;
	nu=0;
	nup=1;
	psioverlaps(nu*sizej+j,nup*sizej+jp)+=wave0(n,j+1)*wave1(n,jp+1)*del_r;
	nu=1;
	nup=0;
	psioverlaps(nu*sizej+j,nup*sizej+jp)+=wave1(n,j+1)*wave0(n,jp+1)*del_r;
	nu=1;
	nup=1;
	psioverlaps(nu*sizej+j,nup*sizej+jp)+=wave1(n,j+1)*wave1(n,jp+1)*del_r;
      }
      for (int nu=0;nu<2;nu++)
      	for (int nup=0;nup<2;nup++)
	  logout<<"overlap <"<<nu<<2*j<<"|"<<nup<<2*jp<<"> = "<<psioverlaps(nu*sizej+j,nup*sizej+jp)<<endl;
    }
  }


  
  //calculate r1,r2 part of potential on the grid

  vector rgrid(size_w);
  matrix wave_r(size_w,sizej*2);
  for (i=0;i<size_w;i++) {
    rgrid(i)=wave0(i,0);
    for (j=0;j<sizej;j++) {
      wave_r(i,0*sizej+j)=wave0(i,j+1);
      wave_r(i,1*sizej+j)=wave1(i,j+1);
    }
  }

  // use dummy theta's and phi
  theta1=0.;
  theta2=0.;
  phi=0.;
  double potl[8]; //array to store pieces of potential
  matrix A000(size_w,size_w);
  matrix A022(size_w,size_w);
  matrix A202(size_w,size_w);
  matrix A224(size_w,size_w);
  for (i=0;i<size_w;i++) {
    r1=rgrid(i)/BohrToA;
    for (j=0;j<size_w;j++) {
      r2=rgrid(j)/BohrToA;  
      vh2h2pieces_(&Rpt,&r1,&r2,&theta1,&theta2,&phi,&V6d,potl);
      A000(i,j)=potl[0];
      A022(i,j)=potl[1];
      A202(i,j)=potl[2];
      A224(i,j)=potl[3];
    }
  }
  // calculate theta1,theta2,phi part on the grid
  // use dummy r's
  r1=6.;
  r2=6.;
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
  // <n1 j1 n2 j2|A|n1' j1' j2'>
  matrix A000_basis(size_nu*sizej*sizej,size_nu*sizej*sizej);
  matrix A022_basis(size_nu*sizej*sizej,size_nu*sizej*sizej);
  matrix A202_basis(size_nu*sizej*sizej,size_nu*sizej*sizej);
  matrix A224_basis(size_nu*sizej*sizej,size_nu*sizej*sizej);
  vector Dr1_000(size_w);
  vector Dr1_022(size_w);
  vector Dr1_202(size_w);
  vector Dr1_224(size_w);

  // j1 and j2 are indices here, not values
  for (int nut=0;nut<size_nu;nut++) {
    n1=nulist1[nut];
    n2=nulist2[nut];
    for (j1=0;j1<sizej;j1++){
      for (j2=0;j2<sizej;j2++){
	for (int nutp=0;nutp<size_nu;nutp++) {	  
	  n1p=nulist1[nutp];
	  n2p=nulist2[nutp];
	  for (j1p=0;j1p<sizej;j1p++){
	    for (j2p=0;j2p<sizej;j2p++){
	      
	      // r1 quadrature
	      for (i=0;i<size_w;i++) {
		Dr1_000(i)=0.;
		Dr1_022(i)=0.;
		Dr1_202(i)=0.;
		Dr1_224(i)=0.;
		//r2 quadrature first
		for (j=0;j<size_w;j++) {
		  Dr1_000(i)+=A000(i,j)*wave_r(j,n2*sizej+j2)*wave_r(j,n2p*sizej+j2p);
		  Dr1_022(i)+=A022(i,j)*wave_r(j,n2*sizej+j2)*wave_r(j,n2p*sizej+j2p);
		  Dr1_202(i)+=A202(i,j)*wave_r(j,n2*sizej+j2)*wave_r(j,n2p*sizej+j2p);
		  Dr1_224(i)+=A224(i,j)*wave_r(j,n2*sizej+j2)*wave_r(j,n2p*sizej+j2p);
		}		 
		// volume elements for quadratures
		Dr1_000(i)*=del_r;
		Dr1_022(i)*=del_r;
		Dr1_202(i)*=del_r;
		Dr1_224(i)*=del_r;		  
		A000_basis((nut*sizej+j1)*sizej+j2,(nutp*sizej+j1p)*sizej+j2p)+=Dr1_000(i)*wave_r(i,n1*sizej+j1)*wave_r(i,n1p*sizej+j1p);
		A022_basis((nut*sizej+j1)*sizej+j2,(nutp*sizej+j1p)*sizej+j2p)+=Dr1_022(i)*wave_r(i,n1*sizej+j1)*wave_r(i,n1p*sizej+j1p);
		A202_basis((nut*sizej+j1)*sizej+j2,(nutp*sizej+j1p)*sizej+j2p)+=Dr1_202(i)*wave_r(i,n1*sizej+j1)*wave_r(i,n1p*sizej+j1p);
		A224_basis((nut*sizej+j1)*sizej+j2,(nutp*sizej+j1p)*sizej+j2p)+=Dr1_224(i)*wave_r(i,n1*sizej+j1)*wave_r(i,n1p*sizej+j1p);

		
	      }
	      A000_basis((nut*sizej+j1)*sizej+j2,(nutp*sizej+j1p)*sizej+j2p)*=del_r;
	      A022_basis((nut*sizej+j1)*sizej+j2,(nutp*sizej+j1p)*sizej+j2p)*=del_r;
	      A202_basis((nut*sizej+j1)*sizej+j2,(nutp*sizej+j1p)*sizej+j2p)*=del_r;
	      A224_basis((nut*sizej+j1)*sizej+j2,(nutp*sizej+j1p)*sizej+j2p)*=del_r;
	    }
	  }
	}
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
      int m2index=-1;
      for (int m2=-jmax;m2<=jmax;m2++) {
	m2index++;
	for (j2=abs(m2);j2<=jmax;j2++){
	  if (j2%2) continue; // skip iteration if j2 odd
	  index_jm12++;
	  int index_jm12p=-1;
	  for (int m1p=-jmax;m1p<=jmax;m1p++) {
	    for (j1p=abs(m1p);j1p<=jmax;j1p++){
	      if (j1p%2) continue; // skip iteration if j1 odd
	      int m2pindex=-1;
	      for (int m2p=-jmax;m2p<=jmax;m2p++) {
		m2pindex++;
		for (j2p=abs(m2p);j2p<=jmax;j2p++){
		  if (j2p%2) continue; // skip iteration if j2 odd
		  index_jm12p++;
		  if ((m1-m2)==(m1p-m2p)){

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
			  Dphi_G000(i*size_theta2+j)+=G000(i*size_theta2+j,k)*basisfunctionP(m2,weights_phi(k),phi)*basisfunctionP(m2p,weights_phi(k),phi);
			  //Dphi_G000(i*size_theta2+j)+=G000(i*size_theta2+j,k)*basisP(m2index,k)*basisP(m2pindex,k);
			  Dphi_G022(i*size_theta2+j)+=G022(i*size_theta2+j,k)*basisfunctionP(m2,weights_phi(k),phi)*basisfunctionP(m2p,weights_phi(k),phi);
			  Dphi_G202(i*size_theta2+j)+=G202(i*size_theta2+j,k)*basisfunctionP(m2,weights_phi(k),phi)*basisfunctionP(m2p,weights_phi(k),phi);
			  Dphi_G224(i*size_theta2+j)+=G224(i*size_theta2+j,k)*basisfunctionP(m2,weights_phi(k),phi)*basisfunctionP(m2p,weights_phi(k),phi);
			}
			Dtheta_G000(i)+=Dphi_G000(i*size_theta2+j)*basisPjm(j2,abs(m2),weights_theta1(j),cos(theta2))*basisPjm(j2p,abs(m2p),weights_theta1(j),cos(theta2));
			Dtheta_G022(i)+=Dphi_G022(i*size_theta2+j)*basisPjm(j2,abs(m2),weights_theta1(j),cos(theta2))*basisPjm(j2p,abs(m2p),weights_theta1(j),cos(theta2));
			Dtheta_G202(i)+=Dphi_G202(i*size_theta2+j)*basisPjm(j2,abs(m2),weights_theta1(j),cos(theta2))*basisPjm(j2p,abs(m2p),weights_theta1(j),cos(theta2));
			Dtheta_G224(i)+=Dphi_G224(i*size_theta2+j)*basisPjm(j2,abs(m2),weights_theta1(j),cos(theta2))*basisPjm(j2p,abs(m2p),weights_theta1(j),cos(theta2));
		      }
		      G000_basis(index_jm12,index_jm12p)+=Dtheta_G000(i)*basisPjm(j1,(m1),weights_theta1(i),cos(theta1))*basisPjm(j1p,(m1p),weights_theta1(i),cos(theta1));
		      G022_basis(index_jm12,index_jm12p)+=Dtheta_G022(i)*basisPjm(j1,(m1),weights_theta1(i),cos(theta1))*basisPjm(j1p,(m1p),weights_theta1(i),cos(theta1));
		      G202_basis(index_jm12,index_jm12p)+=Dtheta_G202(i)*basisPjm(j1,(m1),weights_theta1(i),cos(theta1))*basisPjm(j1p,(m1p),weights_theta1(i),cos(theta1));
		      G224_basis(index_jm12,index_jm12p)+=Dtheta_G224(i)*basisPjm(j1,(m1),weights_theta1(i),cos(theta1))*basisPjm(j1p,(m1p),weights_theta1(i),cos(theta1));
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
      // setup H

  //cout<<size_nu<<" "<<sizej1j2m<<" "<<jmax<<endl;
  matrix H(size_nu*sizej1j2m,size_nu*sizej1j2m);
  matrix V(size_nu*sizej1j2m,size_nu*sizej1j2m);
  matrix delta(size_nu*sizej1j2m,size_nu*sizej1j2m);

  ofstream Vout("V");

  Vout<<" <j1 m1 n1 j2 m2 n2 |V| j1' m1' n1' j2' m2' n2'>"<<endl;
  ofstream Hout("H");

  for (int nut=0;nut<size_nu;nut++) {
    n1=nulist1[nut];
    n2=nulist2[nut];
    for (int nutp=0;nutp<size_nu;nutp++) {
      n1p=nulist1[nutp];
      n2p=nulist2[nutp];
      index_jm12=-1;
      for (int m1=-jmax;m1<=jmax;m1++) {
	for (j1=abs(m1);j1<=jmax;j1++){
	  if (j1%2) continue; // skip iteration if j1 odd
	  for (int m2=-jmax;m2<=jmax;m2++) {
	    for (j2=abs(m2);j2<=jmax;j2++){
	      if (j2%2) continue; // skip iteration if j2 odd
	      index_jm12++;
	      H(nut*sizej1j2m+index_jm12,nut*sizej1j2m+index_jm12)=(Enuj(nulist1[nut]*sizej+j1/2)+Enuj(nulist2[nut]*sizej+j2/2));
	      int index_jm12p=-1;
	      for (int m1p=-jmax;m1p<=jmax;m1p++) {
		for (j1p=abs(m1p);j1p<=jmax;j1p++){
		  if (j1p%2) continue; // skip iteration if j1 odd
		  for (int m2p=-jmax;m2p<=jmax;m2p++) {
		    for (j2p=abs(m2p);j2p<=jmax;j2p++){
		      if (j2p%2) continue; // skip iteration if j2 odd
		      index_jm12p++;		       
		  
		      V(nut*sizej1j2m+index_jm12,nutp*sizej1j2m+index_jm12p)=
			A000_basis((nut*sizej+j1/2)*sizej+j2/2,(nutp*sizej+j1p/2)*sizej+j2p/2)*G000_basis(index_jm12,index_jm12p)
			+A022_basis((nut*sizej+j1/2)*sizej+j2/2,(nutp*sizej+j1p/2)*sizej+j2p/2)*G022_basis(index_jm12,index_jm12p)
			+A202_basis((nut*sizej+j1/2)*sizej+j2/2,(nutp*sizej+j1p/2)*sizej+j2p/2)*G202_basis(index_jm12,index_jm12p)
			+A224_basis((nut*sizej+j1/2)*sizej+j2/2,(nutp*sizej+j1p/2)*sizej+j2p/2)*G224_basis(index_jm12,index_jm12p);

		      double value=V(nut*sizej1j2m+index_jm12,nutp*sizej1j2m+index_jm12p);		  

		      // <j1 m1 n1 j2 m2 n2 |V| j1' m1' n1' j2' m2' n2'>		     
		      if (fabs(value) > 1e-12)
			Vout<<j1<<" "<<m1<<" "<<n1<<"  "<<j2<<" "<<m2<<" "<<n2<<" | "<<j1p<<" "<<m1p<<" "<<n1p<<"  "<<j2p<<" "<<m2p<<" "<<n2p<<" "<<value<<endl;
		      
		      // diagonal part
		      if (j1==j1p && j2==j2p &&  n1==n1p && n2==n2p && m1==(m1p)&& m2==(m2p))
			delta(nut*sizej1j2m+index_jm12,nut*sizej1j2m+index_jm12)+=.5;

		      // P12
		      if (j1==j2p && j2==j1p && m1==(m2p) && m2==(m1p) && n1==n2p && n2==n1p)
			delta(nut*sizej1j2m+index_jm12,nutp*sizej1j2m+index_jm12p)+=.5;
		    }
		  }
		}
	      }
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
  Hout.precision(4);
  Hout.width(4);
  for (int ii=0;ii<size_nu*sizej1j2m;ii++){
    for (int jj=0;jj<size_nu*sizej1j2m;jj++) {
      double val=H(ii,jj)+V(ii,jj);
      if (fabs(val) <1e-10) val=0;
      Hout<<val<<"  ";
    }
    Hout<<endl;
  }
  Hout<<endl;
  
  
  matrix Hsym=delta*(H*delta);
  vector evsym=diag(Hsym);
  vector ev=diag(H);
  time (&diagend);
  double dif2 = difftime (diagend,diagstart);
  sprintf (timemsg, "Elapsed time for diagonalization is %.2lf seconds.", dif2);
  logout<<timemsg<<endl;

  ofstream evalues("ev");
  ofstream evecs("evec");
  for (i=0;i<size_nu*sizej1j2m;i++){
    evalues<<ev(i)<<"   ";
    for (j=0;j<size_nu*sizej1j2m;j++)
      evecs<<H(i,j)<<" ";
     evalues<<endl;
     evecs<<endl;
  }
  ofstream evaluessym("evsym");
  for (i=0;i<size_nu*sizej1j2m;i++){
    evaluessym<<evsym(i)<<"   ";
    for (j=0;j<size_nu*sizej1j2m;j++)
      evaluessym<<Hsym(j,i)<<" ";
     evaluessym<<endl;
  }

  if (size_nu==1) 
    cout<<Rpt*BohrToA<<" "<<ev(0)-E0j(0)-E0j(0)<<" "<<evsym(0)-E0j(0)-E0j(0)<<endl;
  
  if (size_nu==2) 
    cout<<Rpt*BohrToA<<"  "<<ev(0)-Enuj(nulist1[0]*sizej+0)-Enuj(nulist2[0]*sizej+0)<<" "<<evsym(0)-Enuj(nulist1[0]*sizej+0)-Enuj(nulist2[0]*sizej+0)<<endl;

  if (size_nu==4) {
        cout<<Rpt*BohrToA<<" "<<ev(0)-E0j(0)-E0j(0)<<" "<<evsym(0)-E0j(0)-E0j(0)<<endl;
  }
  
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
// ph1
// phi=phi1-phi2
// phi2=phi1-phi

// e^(i m1 phi1)e^(i m2 phi2)=e^(i m1 phi1)e^(i m2 (phi1-phi))=e^(i (m1+m2) phi1)e^(-i m2 phi))
//
// m1+m2=0 restriction so m1=-m2 and we get e^(i (m1+m2) phi1)e^(i m1 phi))
//
// full basis  e^(i (m1+m2) phi1)e^(i m1 phi))  e^(-i (m1p+m2p) phi1)e^(-i m1p phi))
//
// m1+m2+m1p+m2p=0 yields non-zero matrix element
//
// m1+m1p=0 and m2+m2p=0
//
// m1+m2=m+, m2

double PjNormal(int j,int m, double x)
{
  if (j < m) cerr<<"j < m"<<endl;
  double jv=(double)j;
  double mv=fabs((double)m);
  double PjN= sqrt((jv+.5)*exp(lgamma(jv-mv+1.)-lgamma(jv+mv+1.)))*plgndr(j,abs(m),x); // for the normalization constant sqrt(...) refer to 6.55 in Quantum Chemistry. lgamma is the natural logarithim of the gamma function: gamma(n) = (n-1)!
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

matrix fbasisP(int jmax,int size_phi,vector &weights_phi,vector &grid_phi)
{
  matrix basisP(size_phi,2*jmax+1);
  int m,k;
  int indexm=0;
  for (m=-jmax;m<=jmax;m++) {
    for (k=0;k<size_phi;k++) {
      basisP(k,indexm)=basisfunctionP(m,weights_phi(k),grid_phi(k));
    }
    indexm++;
  }
  return basisP;
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
      indexmj++;
    }
  }
  sizes[0]=indexmj; // sizemj
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
