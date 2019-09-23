#include "cmdstuff.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sstream>
#include <iomanip>

using namespace std;

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
double basisPjm(int j,int m,double w,double x);
double vpot(double , double, double );

int main(int argc, char **argv) {
	
    double Rpt          = atof(argv[1]);
    double DipoleMoment = atof(argv[2]);
    double B            = 20.561*CMRECIP2KL;

    int sizej           = 10;
    int jmax            = sizej; //2*(sizej-1);

    int size_theta     = 4.*jmax;
    vector weights_theta(size_theta);
    vector grid_theta = thetagrid(size_theta, weights_theta);

    ofstream logout("log");

    logout<<"  // test orthonormality of theta and phi basis"<<endl;
    for (int j1 = 0; j1 < jmax; j1++)
    {
        for (int j1p = 0; j1p < jmax; j1p++)
        {
            double sum=0.;
	        for (int i = 0; i < size_theta; i++)
            {
                int m= 0;
	            sum += basisPjm(j1,m,weights_theta(i),grid_theta(i))*basisPjm(j1p,m,weights_theta(i),grid_theta(i));
	        }
	    logout<<" j1 "<<j1<<" j1p "<<j1p<<" Norm "<<sum<<endl;
        }
    }

    vector Theta_V(size_theta);
    for (int i = 0; i < size_theta; i++)
    {
        Theta_V(i) = vpot(Rpt, DipoleMoment, grid_theta(i));
    }

    matrix V(jmax,jmax);
    matrix Vd(jmax,jmax);
    double weight, cost;
    int m = 0;
    for (int j1 = 0; j1 < jmax; j1++)
    {
	    for (int j1p = 0; j1p < jmax; j1p++)
        {
            long double sum  = 0.0;
            double sumd = 0.0;
	        for (int i = 0; i < size_theta; i++)
            {
                weight = weights_theta(i);
                cost   = grid_theta(i);
		        sum+= Theta_V(i)*basisPjm(j1,m,weight,cost)*basisPjm(j1p,m,weight,cost);
		        sumd+= grid_theta(i)*basisPjm(j1,m,weight,cost)*basisPjm(j1p,m,weight,cost);
		    }
            V(j1,j1p) = sum;
            Vd(j1,j1p) = sumd;
	    }
    }

    matrix H(jmax,jmax);
    matrix Hrot(jmax,jmax);

    ofstream Vout("V");
    ofstream Hout("H");
/*
    for (int i = 0; i < jmax; i++)
    {
        for (int j = 0; j < jmax; j++)
        {
            if ((abs(V(i,j)) > 10e-18) || (abs(V(j,i)) > 10e-18))
            {
                if (V(i,j) != V(j,i))
                {
                    cout << i<<"   "<<j<<"      "<<V(i,j)<<"        "<<V(j,i)<<endl;
                }
            }
        }
    }
*/
    for (int j1 = 0; j1 < jmax; j1++)
    {
	    H(j1,j1) = B*(double)(j1*(j1+1));
    }

  // Diagonalize

    Hrot = H;
    H = H + V;

/*
    for (int i = 0; i < jmax; i++)
    {
        for (int j = 0; j < jmax; j++)
        {
            if ((abs(H(i,j)) > 10e-12) || (abs(H(j,i)) > 10e-12))
            {
                if (H(i,j) != H(j,i))
                {
                    cout << i<<"   "<<j<<"      "<<H(i,j)<<"        "<<H(j,i)<<endl;
                }
            }
        }
    }
*/
    vector ev=diag(H);

    ofstream evalues("ev");
    evalues.precision(3);
    for (int i = 0; i < jmax; i++)
    {
        evalues<<ev(i)<<"   ";
        for (int j = 0; j < jmax; j++)
		{
            evalues<<H(j,i)<<" ";
        }
        evalues<<endl;
    }

	matrix avgV    = transpose(H)*V*H;
	matrix avgHrot = transpose(H)*Hrot*H;
	matrix avgVd   = transpose(H)*Vd*H;
    logout<<"B     = "<<B<<" cm-1"<<endl;
    logout<<"E_0   = "<<ev(0)<<" cm-1"<<endl;

    stringstream bc;
    bc.width(4);
    bc.fill('0');
    bc<<DipoleMoment;
    stringstream rpt;
    //rpt.width(4);
    //rpt.fill('0');
    rpt<<Rpt;
    string fname = "EigenValuesFor1HF-Rpt" + rpt.str() + "Angstrom-DipoleMoment" + bc.str()+"Debye.txt";

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
    eigenvalues<< setw(20)<<setfill(' ')<<avgHrot(0,0);
    eigenvalues<< setw(20)<<setfill(' ')<<avgVd(0,0)<<endl;

    double uvec1[jmax];
    double psi[size_theta];
    for (int i = 0; i < size_theta; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < jmax; j++)
        {
            uvec1[j] = H(j,0); 
            int m = 0;
            sum += uvec1[j]*basisPjm(j,m,weights_theta(i),grid_theta(i))/sqrt(weights_theta(i));
        }
        psi[i] = sum;
    }

    string fname1 = "DataForHistogramFor1HF-Rpt" + rpt.str() + "Angstrom-DipoleMoment" + bc.str()+"Debye.txt";

    ofstream histogram(fname1.c_str());
    histogram.precision(10);
    histogram.setf(ios::right);
    histogram << showpoint;
    histogram << setw(20)<<setfill(' ');

    double sum = 0.0;
    double density[size_theta];
    for (int i = 0; i < size_theta; i++)
    {
        sum += Theta_V(i)*psi[i]*psi[i]*weights_theta(i);
        density[i] = psi[i]*psi[i];
        histogram<< setw(20)<<setfill(' ')<<grid_theta(i);
        histogram<< setw(20)<<setfill(' ')<<psi[i]*psi[i]<<endl;
    }
    cout << sum<<endl;
    exit(0);
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
    grid(i)=x[i];
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
  //double PjN= pow(-1.,mv)*sqrt((jv+.5)*exp(lgamma(jv-mv+1.)-lgamma(jv+mv+1.)))*plgndr(j,abs(m),x); // for the normalization constant sqrt(...) refer to 6.55 in Quantum Chemistry. lgamma is the natural logarithim of the gamma function: gamma(n) = (n-1)!
  double PjN= sqrt((jv+.5)*exp(lgamma(jv-mv+1.)-lgamma(jv+mv+1.)))*plgndr(j,abs(m),x); // for the normalization constant sqrt(...) refer to 6.55 in Quantum Chemistry. lgamma is the natural logarithim of the gamma function: gamma(n) = (n-1)!
  //if (m<0)
   // PjN=pow(-1.,mv)*PjN; // refer to pg. 9 Angular Momentum, Richard N. Zare 
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

    if (m>0)
    { 
        phase=pow((-1.),(m)); 
    }
    else
    {
        phase=1.;							
    }
    value=sqrt(w)*phase*PjNormal(j,m,x);
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

double vpot(double Rpt, double DipoleMoment, double Theta)
{
    double DipoleMomentAu = DipoleMoment/AuToDebye;
    double RptAu          = Rpt/AuToAngstrom;
    double PotAu          = -2.0*DipoleMomentAu*DipoleMomentAu*Theta/(RptAu*RptAu*RptAu);
    return PotAu*AuToKelvin;
}
