#include "cmdstuff.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sstream>
#include <iomanip>

using namespace std;

static const double hatocm=219474.63067;
static const double AuToAngstrom  = 0.52917720859;
static const double AuToDebye     = 1.0/0.39343;
static const double AuToCmInverse = 219474.63137;
static const double AuToKelvin    = 315777.0;
static const double CMRECIP2KL    = 1.4387672;

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
double PotFunc(double, double, double, double, double);
double RelativeAngle(double, double, double);

int main(int argc,char **argv) {

    double Rpt          = atof(argv[1]);
    double DipoleMoment = atof(argv[2]);
    int sizej           = atoi(argv[3]);

    double B            = 20.9561*CMRECIP2KL;

    int jmax            = 2*(sizej-1);
    int size_theta1     = 2.*jmax+5;
    int size_theta2     = 2.*jmax+5;
    int size_phi        = 2.*(2.*jmax+7); // 

    vector weights_theta1(size_theta1);
    vector weights_theta2(size_theta2);
    vector weights_phi(size_phi); 

    vector grid_theta1  = thetagrid(size_theta1,weights_theta1);
    vector grid_theta2  = thetagrid(size_theta2,weights_theta2);
    vector grid_phi     = phigrid(size_phi,weights_phi);


    ofstream logout("log");

    int skip = 1;
    logout<<"  // test orthonormality of theta and phi basis"<<endl;
    for (int m1 = -jmax; m1 <= jmax; m1++) 
    {
        for (int j1 = abs(m1); j1 <= jmax; j1++)
        {
            if (j1%skip) continue; 
            int m2 = m1;
            for (int j2 = abs(m2); j2 <= jmax; j2++)
            {
	            if (j2%skip) continue; 
	            double sum = 0.;
	            for (int i = 0; i < size_theta1; i++)
                {
	                double theta1 = grid_theta1(i);
	                sum += basisPjm(j1,m1,weights_theta1(i),cos(theta1))*basisPjm(j2,m2,weights_theta1(i),cos(theta1));
	            }
	            logout<<" m1 = "<<m1<<" "<<j1<<" "<<j2<<" "<<sum<<endl;
            }
        }
    }

	for (int m1 = -jmax; m1 <= jmax; m1++) 
	{
    	for (int m1p = -jmax; m1p <= jmax; m1p++)
		{
      		double sum = 0.;
      		for (int k = 0; k < size_phi; k++)
			{
				double phi = grid_phi(k);
				sum       +=basisfunctionP(m1,weights_phi(k),phi)*basisfunctionP(m1p,weights_phi(k),phi);
      		}
      		logout<<" m1,m1p = "<<m1<<" "<<m1p<<" "<<sum<<endl;      
    	}
  	}
 
  // **************************************
  // Define kinetic energy info
  // **************************************


    int sizes[2];
    get_sizes(jmax,sizes);
    int sizej1j2m=sizes[1];

	matrix Cost(size_theta1*size_theta2, size_phi);
	matrix VV(size_theta1*size_theta2,size_phi);
	for (int i = 0; i < size_theta1; i++)
    {
    	double theta1 = grid_theta1(i);
        for (int j = 0; j < size_theta2; j++)
        {
            double theta2 = grid_theta2(j);
            for (int k = 0; k < size_phi; k++)
            {
	            double phi = grid_phi(k);
	            VV(i*size_theta2+j,k)    = PotFunc(Rpt, DipoleMoment, theta1, theta2, phi);
                Cost(i*size_theta2+j, k) = RelativeAngle(theta1, theta2, phi);
            }
        }
	}

	vector Dphi_VV(size_theta1*size_theta2);
	vector Dtheta_VV(size_theta1);
	matrix VV_basis(sizej1j2m,sizej1j2m);

	vector Dphi_Cost(size_theta1*size_theta2);
	vector Dtheta_Cost(size_theta1);
	matrix Cost_basis(sizej1j2m,sizej1j2m);
  
	int index_jm12 = 0;
	for (int m1 = -jmax; m1 <= jmax; m1++) 
    {
    	for (int j1 = abs(m1); j1 <= jmax; j1++)
		{
      		if (j1%skip) continue; 
      		int m2 = -m1;

      		for (int j2 = abs(m2); j2 <= jmax; j2++)
			{
				if (j2%skip) continue;

				int index_jm12p = 0;
				for (int m1p =- jmax; m1p <= jmax; m1p++) 
				{
	  				for (int j1p = abs(m1p); j1p <= jmax; j1p++)
					{
	    				if (j1p%skip) continue;

	    				int m2p = -m1p;
	    				for (int j2p = abs(m2p); j2p <= jmax; j2p++)
						{
	      					if (j2p%skip) continue; 

	      					for (int i = 0; i < size_theta1; i++)
							{
								double theta1 = grid_theta1(i);
								Dtheta_VV(i) = 0.;
								Dtheta_Cost(i) = 0.;

								for (int j = 0; j < size_theta2; j++)
								{
		  							double theta2 = grid_theta2(j);
		  							Dphi_VV(i*size_theta2+j) = 0.;
		  							Dphi_Cost(i*size_theta2+j) = 0.;
		  							for (int k = 0; k < size_phi; k++)
									{
		    							double phi = grid_phi(k);
	Dphi_VV(i*size_theta2+j) += VV(i*size_theta2+j,k)*basisfunctionP(m1,weights_phi(k),phi)*basisfunctionP(m1p,weights_phi(k),phi);
	Dphi_Cost(i*size_theta2+j) += Cost(i*size_theta2+j,k)*basisfunctionP(m1,weights_phi(k),phi)*basisfunctionP(m1p,weights_phi(k),phi);

		  							}

	Dtheta_VV(i) += Dphi_VV(i*size_theta2+j)*basisPjm(j2,m2,weights_theta1(j),cos(theta2))*basisPjm(j2p,m2p,weights_theta1(j),cos(theta2));
	Dtheta_Cost(i) += Dphi_Cost(i*size_theta2+j)*basisPjm(j2,m2,weights_theta1(j),cos(theta2))*basisPjm(j2p,m2p,weights_theta1(j),cos(theta2));

								}

	VV_basis(index_jm12,index_jm12p)+=Dtheta_VV(i)*basisPjm(j1,abs(m1),weights_theta1(i),cos(theta1))*basisPjm(j1p,abs(m1p),weights_theta1(i),cos(theta1));
	Cost_basis(index_jm12,index_jm12p)+=Dtheta_Cost(i)*basisPjm(j1,abs(m1),weights_theta1(i),cos(theta1))*basisPjm(j1p,abs(m1p),weights_theta1(i),cos(theta1));

	      					}
	      					index_jm12p++;
	    				}
	  				}
				}
				index_jm12++;
      		}
    	}
  	}
    
  	matrix H(sizej1j2m,sizej1j2m);
  	matrix R(sizej1j2m,sizej1j2m);
  	matrix V(sizej1j2m,sizej1j2m);

  	ofstream Vout("V");
  	ofstream Hout("H");

  	index_jm12 = 0;
  	for (int m1 = -jmax; m1 <= jmax; m1++) 
    {
    	for (int j1 = abs(m1); j1 <= jmax; j1++)
        {
      		if (j1%skip) continue;

      		int m2 = -m1;
      		for (int j2 = abs(m2); j2 <= jmax; j2++)
            {
				if (j2%skip) continue; 

				H(index_jm12,index_jm12) = B*((double)(j1*(j1+1))+(double)(j2*(j2+1)));

				int index_jm12p = 0;
				for (int m1p = -jmax; m1p <= jmax; m1p++) 
                {
	  				for (int j1p = abs(m1p); j1p <= jmax; j1p++)
                    {
	    				if (j1p%skip) continue;

	    				int m2p = -m1p;
	    				for (int j2p = abs(m2p); j2p <= jmax; j2p++)
                        {
	      					if (j2p%skip) continue; 
	      
	      					V(index_jm12,index_jm12p) = VV_basis(index_jm12,index_jm12p);
	      					index_jm12p++;		       
	    				}
	  				}
				}
				index_jm12++;
      		}	
    	}
  	}

 	R = H;
  	H = H + V;

  	//  ofstream Hout("H");
	for (int i = 0; i < sizej1j2m; i++)
	{
    	for (int j = 0; j < sizej1j2m; j++)
            Hout<<H(i,j)<<" ";
        Hout<<endl;
    }
  
  	vector ev = diag(H);

  	ofstream evalues("ev");
  	evalues.precision(3);
  	for (int i = 0; i < sizej1j2m; i++)
	{
    	evalues<<ev(i)<<"   ";
    	for (int j = 0; j < sizej1j2m; j++)
      	    evalues<<H(j,i)<<" ";
     	evalues<<endl;
  	}

	matrix avgV=	transpose(H)*V*H;
	matrix avgR=	transpose(H)*R*H;
	matrix avgCost=	transpose(H)*Cost_basis*H;

    stringstream bc;
    bc.width(4);
    bc.fill('0');
    bc<<DipoleMoment;
    string fname = "EigenValuesFor2HF-DipoleMoment" + bc.str()+".txt";

    //ofstream eigenvalues;
    //eigenvalues.open(fname.c_str(), ios::app);

    ofstream eigenvalues(fname.c_str(), ios::app);
    eigenvalues.precision(10);
    eigenvalues.setf(ios::right);
    eigenvalues << showpoint;
    eigenvalues << setw(20)<<setfill(' ');
    eigenvalues<< setw(20)<<setfill(' ')<<Rpt;
    eigenvalues<< setw(20)<<setfill(' ')<<ev(0);
    eigenvalues<< setw(20)<<setfill(' ')<<(ev(1) - ev(0));
    eigenvalues<< setw(20)<<setfill(' ')<<avgV(0,0);
    eigenvalues<< setw(20)<<setfill(' ')<<avgR(0,0);
    eigenvalues<< setw(20)<<setfill(' ')<<avgCost(0,0)<<endl;
    exit(0);
}


vector thetagrid(int nsize,vector &weights)
{
	vector grid(nsize);
	double *x = new double[nsize];
	double *w = new double[nsize];
  	double x1 = -1.;
  	double x2 = 1.;
  	gauleg(x1,x2,x,w,nsize);
  	for (int i = 0; i < nsize; i++) 
	{
    	grid(i) = acos(x[i]);
    	weights(i) = w[i]; 
    }
    return grid;
}

vector phigrid(int nsize,vector &weights)
{
    vector grid(nsize);

    double phimin=0.*M_PI;
    double phimax=2.*M_PI;
    double dphi=2.*M_PI/(double)nsize;
    for (int i = 0; i < nsize; i++) 
    {
        grid(i) = phimin+((double)i)*dphi;
        weights(i) = dphi;
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
							
      if (j1%1) continue; // skip iteration if j1 odd
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
							
      if (j2%1) continue; // skip iteration if j2 odd
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
							
      if (j1%1) continue; // skip iteration if j odd

      int indexmj2_in=0;
      for (j2=abs(m);j2<=jmax;j2++) {

	if (j2%1) continue;
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
double PotFunc(double Rpt, double DipoleMoment, double theta1, double theta2, double phi)
{
    double DipoleMomentAu = DipoleMoment/AuToDebye;
    double RptAu          = Rpt/AuToAngstrom;

    double PotAu          = DipoleMomentAu*DipoleMomentAu*(sin(theta1)*sin(theta2)*cos(phi) - 2.0*cos(theta1)*cos(theta2))/(RptAu*RptAu*RptAu);
    return PotAu*AuToKelvin;
}

double RelativeAngle(double theta1, double theta2, double phi)
{
    double Angle          = sin(theta1)*sin(theta2)*cos(phi) + cos(theta1)*cos(theta2);
    return Angle;
}
