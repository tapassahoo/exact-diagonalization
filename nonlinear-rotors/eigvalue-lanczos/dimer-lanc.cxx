#include "cmdstuff.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include <sstream>
#include <iomanip>
#include "random.h"
//#include "BF.h"

using namespace std;

static const double AuToAngstrom  = 0.52917720859;
static const double AuToDebye     = 1.0/0.39343;
static const double AuToCmInverse = 219474.63137;
static const double AuToKelvin    = 315777.0;
static const double CMRECIP2KL    = 1.4387672;
static const double pi=3.141592653589793;

void lancbis(int niter,vector &eval,vector &evalerr,double elmin,
double elmax,int &ngood,const vector& alpha,const vector& beta,
const vector& beta2);
EXTERNC void gauleg(double x1,double x2,double *x,double *w,int n);
EXTERNC double plgndr(int j,int m,double x);
vector thetagrid(int nsize,vector &weights);
vector phigrid(int nsize,vector &weights);
void get_sizes(int jmax, int *sizes);
void get_QuantumNumList(int jmax, matrix &jkemQuantumNumList, matrix &jkomQuantumNumList);
double off_diag(int j, int k);
double littleD(int ldJ, int ldmp, int ldm, double ldtheta);
void get_HrotN2(double ah2o, double bh2o, double ch2o, int jkem, matrix &jkemQuantumNumList, matrix &HrotKee);
void get_Hrot(double ah2o, double bh2o, double ch2o, int jkem, matrix &jkemQuantumNumList, vector &HrotKe);
void get_indvbasis(int jkem, int size_theta, vector &grid_theta, vector &weights_theta, int size_phi, vector &grid_phi, vector &weights_phi, matrix &jkemQuantumNumList, matrix &dJKeM, matrix &KJKeMRe, matrix &KJKeMIm, matrix &MJKeMRe, matrix &MJKeMIm);
void check_norm(int jkem, int size_theta, int size_phi, matrix &dJKeM, matrix &KJKeMRe, matrix &KJKeMIm, matrix &MJKeMRe, matrix &MJKeMIm, matrix &jkemQuantumNumList);
void get_JKMbasis(int jkem, int size_theta, int size_phi, matrix &dJKeM, matrix &KJKeMRe, matrix &KJKeMIm, matrix &MJKeMRe, matrix &MJKeMIm, vector &basisJKeMRe, vector &basisJKeMIm);
void check_normJKeM(int jkem, int size_grid, vector &basisJKeMRe, vector &basisJKeMIm, matrix &jkemQuantumNumList); 
void get_pot(int size_theta, int size_phi, vector &grid_theta, vector &grid_phi, double *com1, double *com2, matrix &Vpot);
extern "C" void caleng_(double *com1, double *com2, double *E_2H2O, double *Eulang1, double *Eulang2);
cvector Hv(int jkem, int size_grid, vector &HrotKe, matrix &Vpot, vector &basisJKeMRe, vector &basisJKeMIm, cvector &v0);
void printIndex(int jkem);
 
int main(int argc,char **argv) 
{
    time_t totalstart,totalend,callpotstart,callpotend,diagstart,diagend;
    time (&totalstart);
    char timemsg[100];
	
    double z = atof(argv[1]);
    int sizej = atoi(argv[2]);
	int niter = atoi(argv[3]);
    int jmax  = sizej;//2*(sizej-1);

	//The rotational A, B, C constants are indicated by ah2o, bh2o and ch2o, respectively. The unit is cm^-1.
    double ah2o= 27.877;//cm-1
    double bh2o= 14.512;//cm-1
    double ch2o= 9.285; //cm-1
    ah2o=ah2o*CMRECIP2KL;
    bh2o=bh2o*CMRECIP2KL;
    ch2o=ch2o*CMRECIP2KL;

    ofstream logout("log");

    int size_theta = 2.*jmax+5;
    int size_phi   = 2.*(2.*jmax+7);
    size_theta = jmax+5;
    size_phi   = (2.*jmax+1);

    vector weights_theta(size_theta);
    vector weights_phi(size_phi); 

    vector grid_theta = thetagrid(size_theta, weights_theta);
    vector grid_phi   = phigrid(size_phi, weights_phi);

    // evaluate basis fxns on grid points
    int sizes[3];
    get_sizes(jmax, sizes);
	int jkm=sizes[0];
	int jkem=sizes[1];
	int jkom=sizes[2];

	matrix jkemQuantumNumList(jkem,3);
	matrix jkomQuantumNumList(jkom,3);
	get_QuantumNumList(jmax, jkemQuantumNumList, jkomQuantumNumList);

	vector HrotKe(jkem*jkem);
	get_Hrot(ah2o, bh2o, ch2o, jkem, jkemQuantumNumList, HrotKe);
	
	matrix dJKeM(jkem,size_theta);
	matrix KJKeMRe(jkem,size_phi);
	matrix KJKeMIm(jkem,size_phi);
	matrix MJKeMRe(jkem,size_phi);
	matrix MJKeMIm(jkem,size_phi);
	get_indvbasis(jkem, size_theta, grid_theta, weights_theta, size_phi, grid_phi, weights_phi, jkemQuantumNumList, dJKeM, KJKeMRe, KJKeMIm, MJKeMRe, MJKeMIm);
	//check_norm(jkem, size_theta, size_phi, dJKeM, KJKeMRe, KJKeMIm, MJKeMRe, MJKeMIm, jkemQuantumNumList);
	int size_grid = size_theta*size_phi*size_phi;
	vector basisJKeMRe(jkem*size_grid);
	vector basisJKeMIm(jkem*size_grid);
	get_JKMbasis(jkem, size_theta, size_phi, dJKeM, KJKeMRe, KJKeMIm, MJKeMRe, MJKeMIm, basisJKeMRe, basisJKeMIm);
	check_normJKeM(jkem, size_grid, basisJKeMRe, basisJKeMIm, jkemQuantumNumList); 

	double com1[]={0.0, 0.0, 0.0};
	double com2[]={0.0, 0.0, z};

	matrix Vpot(size_grid, size_grid);
	get_pot(size_theta, size_phi, grid_theta, grid_phi, com1, com2, Vpot);

	//Lanczos
	Rand *randomseed = new Rand(1);

	// PI lanczos
	int ngood;

	cvector r(jkem*jkem);
	vector evalerr(niter);
	vector eval(niter);
	vector alpha(niter);
	vector beta(niter+1);
	vector beta2(niter+1);

	cvector v0(jkem*jkem);
	for (int i=0; i<jkem; i++) {
		for (int j=0; j<jkem; j++) {
			v0(j+i*jkem)=2.0*(randomseed->frand()-0.5);
		}
	}
	cnormalise(v0);

	cvector u=v0;
	logout<<"start iterations"<<endl;
	for (int i=1;i<=niter;i++) {

		u=Hv(jkem, size_grid, HrotKe, Vpot, basisJKeMRe, basisJKeMIm, v0);

		r=r+u;

		alpha(i-1)=real(v0*r);
		r=r-(complex(alpha(i-1),0)*v0);

		beta2(i)=real(r*r);
		beta(i)=sqrt(beta2(i));
		r=complex((1./beta(i)),0)*r; // to get v
		v0=complex((-beta(i)),0)*v0; // prepare r check minus sign!!!

		u=v0;     // swapping
		v0=r;
		r=u;
		if (i%1 == 0) logout<<"iteration "<<i<<endl;
	}                  

	double emax=50.0;
	double emin=-20.0;
	lancbis(niter,eval,evalerr,emin,emax,ngood,alpha,beta,beta2);
	cout<<" ngood = "<<ngood<<endl;
	cout<<"E0 = "<<eval(0)<<endl;
	// lanczos report:
	ofstream lancout("boundstates.out");
	ofstream lanczpeout("states_zpe.out");
	for (int i=0; i<ngood; i++) {
		lancout<<eval(i)<<" "<<evalerr(i)<<endl;
		lanczpeout<<(eval(i)-eval(0))<<endl;
	}
	lancout.flush();
	lancout.close();

	logout.close();
	exit(10);
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

void get_sizes(int jmax, int *sizes)
{
	int jkm=int(((2*jmax+1)*(2*jmax+2)*(2*jmax+3)/6)); //JKM = "Sum[(2J+1)**2,{J,0,Jmax}]" is computed in mathematica

	int jkem, jkom;

	if (jmax%2==0)
	{
		jkem=int((jkm+jmax+1)/2);
		jkom=jkm-jkem;
	}
	else
	{
		jkom=int((jkm+jmax+1)/2);
		jkem=jkm-jkom;
	}

	sizes[0]=jkm;
	sizes[1]=jkem;
	sizes[2]=jkom;

	int jkeem=jkem*jkem;
	int jkeom=jkem*jkom;
	int jkoem=jkom*jkem;
	int jkoom=jkom*jkom;

	int chkjkm = jkeem+jkeom+jkoem+jkoom;
	int jkm2 = jkm*jkm;

    ofstream logout("log");
	if (chkjkm != jkm2)
	{
		logout<<"Wrong index estimation ..."<<endl;
		exit(1);
	}
    
	logout<<"|------------------------------------------------"<<endl;
	logout<<"| Number of basis functions calculations ...."<<endl;
	logout<<"| "<<endl;
	logout<<"| # of |JKM> basis = "<<jkm<<endl;
	logout<<"| "<<endl;
	logout<<"| # of even K in |JKM> = "<<jkem<<endl;
	logout<<"| # of odd  K in |JKM> = "<<jkom<<endl;
	logout<<"| "<<endl;
	logout<<"| # of even K1, even K2 in the |J1K1M1,J2K2M2> = "<<jkeem<<endl;
	logout<<"| # of even K1, odd  K2 in the |J1K1M1,J2K2M2> = "<<jkeom<<endl;
	logout<<"| # of odd  K1, even K2 in the |J1K1M1,J2K2M2> = "<<jkoem<<endl;
	logout<<"| # of odd  K1, odd  K2 in the |J1K1M1,J2K2M2> = "<<jkoom<<endl;
	logout<<"| "<<endl;
	logout<<"| # of |J1K1M1;J2K2M2> basis= # of ChkJKM"<<endl;
	logout<<"| # of |J1K1M1;J2K2M2> basis= "<<jkm2<<endl;
	logout<<"| # of ChkJKM = "<<chkjkm<<endl;
	logout<<"|------------------------------------------------"<<endl;
	logout.close();
}

void get_QuantumNumList(int jmax, matrix &jkemQuantumNumList, matrix &jkomQuantumNumList)
{
	//even K
	int jtempcounter=0;
	for (int j=0; j<(jmax+1); j++)
	{
		if (j%2==0)
		{
			for (int k=-j; k<(j+1); k+=2)
			{
				for (int m=-j; m<(j+1); m++)
				{
					jkemQuantumNumList(jtempcounter,0)=j;
					jkemQuantumNumList(jtempcounter,1)=k;
					jkemQuantumNumList(jtempcounter,2)=m;
					jtempcounter++;
				}
			}
		}
		else
		{
			for (int k=(-j+1); k<j; k+=2)
			{
				for (int m=-j; m<(j+1); m++)
				{
					jkemQuantumNumList(jtempcounter,0)=j;
					jkemQuantumNumList(jtempcounter,1)=k;
					jkemQuantumNumList(jtempcounter,2)=m;
					jtempcounter++;
				}
			}
		}
	}
	//cout<<"jtempcounter "<<jtempcounter<<endl;
    //odd K
	/*
	jtempcounter=0;
	for (int j=0; j<(jmax+1); j++)
	{
		if (j%2==0)
		{
			for (int k=(-j+1); k<j; k+=2)
			{
				for (int m=-j; m<(j+1); m++)
				{
					jkomQuantumNumList(jtempcounter,0)=j;
					jkomQuantumNumList(jtempcounter,1)=k;
					jkomQuantumNumList(jtempcounter,2)=m;
					jtempcounter++;
				}
			}
		}
		else
		{
			for (int k=-j; k<(j+1); k+=2)
			{
				for (int m=-j; m<(j+1); m++)
				{
					jkomQuantumNumList(jtempcounter,0)=j;
					jkomQuantumNumList(jtempcounter,1)=k;
					jkomQuantumNumList(jtempcounter,2)=m;
					jtempcounter++;
				}
			}
		}
	}
	*/
}

double off_diag(int j, int k)
{                     
	/*
	off diagonal <JKM|H|J'K'M'> #
	*/
	double val=sqrt((j*(j+1))-(k*(k+1)));
	return val;
}

double littleD(int ldJ, int ldmp, int ldm, double ldtheta)
{
	/*
	Computation of d(j, m', m, theta) i.e., little d-rotation matrix 
	*/
	double teza =tgamma((ldJ+ldm)+1)*tgamma((ldJ-ldm)+1)*tgamma((ldJ+ldmp)+1)*tgamma((ldJ-ldmp)+1);
	double dval = sqrt(teza);
	double tempD = 0.0;

	//determine max v that will begin to give negative factorial arguments
	int upper;
	if ((ldJ-ldmp) > (ldJ+ldm))
	{
		upper=ldJ-ldmp;
	}
	else
	{
		upper=ldJ+ldm;
	}

	//iterate over intergers that provide non-negative factorial arguments
	for (int v=0; v<(upper+1); v++)
	{
		int a=ldJ-ldmp-v;
		int b=ldJ+ldm-v;
		int c=v+ldmp-ldm;
		if ((a>=0) && (b>=0) && (c>=0))
		{
			tempD += (pow(-1.0, v)/(tgamma(a+1)*tgamma(b+1)*tgamma(c+1)*tgamma(v+1)))*pow(cos(ldtheta/2.0),(2.0*ldJ+ldm-ldmp-2.0*v))*pow(-sin(ldtheta/2.0),(ldmp-ldm+2.0*v));
		}
	}
	return dval*tempD;
}

void get_Hrot(double ah2o, double bh2o, double ch2o, int jkem, matrix &jkemQuantumNumList, vector &HrotKe)
{
    //Computation of Hrot (Asymmetric Top Hamiltonian in Symmetric Top Basis)
	int jkm1, jkmp1, ij;
	for (jkm1=0; jkm1<jkem; jkm1++) {
		for (jkmp1=0; jkmp1<jkem; jkmp1++) {
			ij = jkmp1+jkm1*jkem;

			if ((jkemQuantumNumList(jkm1,0)==jkemQuantumNumList(jkmp1,0)) && (jkemQuantumNumList(jkm1,2)==jkemQuantumNumList(jkmp1,2)))
			{
				if (jkemQuantumNumList(jkm1,1)==(jkemQuantumNumList(jkmp1,1)-2))
				{
					HrotKe(ij) += 0.25*(ah2o-ch2o)*off_diag(jkemQuantumNumList(jkm1,0),jkemQuantumNumList(jkm1,1))*off_diag(jkemQuantumNumList(jkm1,0),jkemQuantumNumList(jkm1,1)+1);
				}
				else if (jkemQuantumNumList(jkm1,1)==(jkemQuantumNumList(jkmp1,1)+2))
				{
					HrotKe(ij) += 0.25*(ah2o-ch2o)*off_diag(jkemQuantumNumList(jkm1,0),jkemQuantumNumList(jkm1,1)-1)*off_diag(jkemQuantumNumList(jkm1,0),jkemQuantumNumList(jkm1,1)-2);
				}
				else if (jkemQuantumNumList(jkm1,1)==(jkemQuantumNumList(jkmp1,1)))
				{
					HrotKe(ij) += (0.5*(ah2o + ch2o)*(jkemQuantumNumList(jkm1,0)*(jkemQuantumNumList(jkm1,0)+1)) + (bh2o - 0.5*(ah2o+ch2o)) * pow(jkemQuantumNumList(jkm1,1),2));
				}
			}
		}
	}
}

void get_HrotN2(double ah2o, double bh2o, double ch2o, int jkem, matrix &jkemQuantumNumList, matrix &HrotKee)
{
    //Computation of Hrot (Asymmetric Top Hamiltonian in Symmetric Top Basis)
	long int jkm12 = 0;
	for (int jkm1=0; jkm1<jkem; jkm1++)
	{
		for (int jkm2=0; jkm2<jkem; jkm2++)
		{
			long int jkmp12 = 0;
			for (int jkmp1=0; jkmp1<jkem; jkmp1++)
			{
				for (int jkmp2=0; jkmp2<jkem; jkmp2++)
				{
					//For 1st rotor
					if ((jkemQuantumNumList(jkm2,0)==jkemQuantumNumList(jkmp2,0)) && (jkemQuantumNumList(jkm2,1)==jkemQuantumNumList(jkmp2,1)) && (jkemQuantumNumList(jkm2,2)==jkemQuantumNumList(jkmp2,2)))
					{
						if ((jkemQuantumNumList(jkm1,0)==jkemQuantumNumList(jkmp1,0)) && (jkemQuantumNumList(jkm1,2)==jkemQuantumNumList(jkmp1,2)))
						{
							if (jkemQuantumNumList(jkm1,1)==(jkemQuantumNumList(jkmp1,1)-2))
							{
								HrotKee(jkm12,jkmp12) += 0.25*(ah2o-ch2o)*off_diag(jkemQuantumNumList(jkm1,0),jkemQuantumNumList(jkm1,1))*off_diag(jkemQuantumNumList(jkm1,0),jkemQuantumNumList(jkm1,1)+1);
							}
							else if (jkemQuantumNumList(jkm1,1)==(jkemQuantumNumList(jkmp1,1)+2))
							{
								HrotKee(jkm12,jkmp12) += 0.25*(ah2o-ch2o)*off_diag(jkemQuantumNumList(jkm1,0),jkemQuantumNumList(jkm1,1)-1)*off_diag(jkemQuantumNumList(jkm1,0),jkemQuantumNumList(jkm1,1)-2);
							}
							else if (jkemQuantumNumList(jkm1,1)==(jkemQuantumNumList(jkmp1,1)))
							{
								HrotKee(jkm12,jkmp12) += (0.5*(ah2o + ch2o)*(jkemQuantumNumList(jkm1,0)*(jkemQuantumNumList(jkm1,0)+1)) + (bh2o - 0.5*(ah2o+ch2o)) * pow(jkemQuantumNumList(jkm1,1),2));
							}
						}
					}

					//For 2nd rotor
					if ((jkemQuantumNumList(jkm1,0)==jkemQuantumNumList(jkmp1,0)) && (jkemQuantumNumList(jkm1,1)==jkemQuantumNumList(jkmp1,1)) && (jkemQuantumNumList(jkm1,2)==jkemQuantumNumList(jkmp1,2)))
					{
						if ((jkemQuantumNumList(jkm2,0)==jkemQuantumNumList(jkmp2,0)) && (jkemQuantumNumList(jkm2,2)==jkemQuantumNumList(jkmp2,2)))
						{
							if (jkemQuantumNumList(jkm2,1)==(jkemQuantumNumList(jkmp2,1)-2))
							{
								HrotKee(jkm12,jkmp12) += 0.25*(ah2o-ch2o)*off_diag(jkemQuantumNumList(jkm2,0),jkemQuantumNumList(jkm2,1))*off_diag(jkemQuantumNumList(jkm2,0),jkemQuantumNumList(jkm2,1)+1);
							}
							else if (jkemQuantumNumList(jkm2,1)==(jkemQuantumNumList(jkmp2,1)+2))
							{
								HrotKee(jkm12,jkmp12) += 0.25*(ah2o-ch2o)*off_diag(jkemQuantumNumList(jkm2,0),jkemQuantumNumList(jkm2,1)-1)*off_diag(jkemQuantumNumList(jkm2,0),jkemQuantumNumList(jkm2,1)-2);
							}
							else if (jkemQuantumNumList(jkm2,1)==(jkemQuantumNumList(jkmp2,1)))
							{
								HrotKee(jkm12,jkmp12) += (0.5*(ah2o + ch2o)*(jkemQuantumNumList(jkm2,0)*(jkemQuantumNumList(jkm2,0)+1)) + (bh2o - 0.5*(ah2o+ch2o)) *pow(jkemQuantumNumList(jkm2,1),2));
							}
						}
					}
					jkmp12++;
				}
			}
			jkm12++;
		}
	}
}

void get_indvbasis(int jkem, int size_theta, vector &grid_theta, vector &weights_theta, int size_phi, vector &grid_phi, vector &weights_phi, matrix &jkemQuantumNumList, matrix &dJKeM, matrix &KJKeMRe, matrix &KJKeMIm, matrix &MJKeMRe, matrix &MJKeMIm)
{
	//block for construction of individual basis begins 
	double nk=1.0;
	for (int s=0; s<jkem; s++)
	{
		for (int th=0; th<size_theta; th++)
		{
			dJKeM(s,th) = sqrt((2.0*jkemQuantumNumList(s,0)+1.0)/(8.0*pow(pi,2)))*littleD(jkemQuantumNumList(s,0),jkemQuantumNumList(s,2),jkemQuantumNumList(s,1),grid_theta(th))*sqrt(weights_theta(th));
		}

		for (int ph=0; ph<size_phi; ph++)
		{
			KJKeMRe(s,ph) = cos(grid_phi(ph)*jkemQuantumNumList(s,1))*sqrt(weights_phi(ph))*nk;
			KJKeMIm(s,ph) = sin(grid_phi(ph)*jkemQuantumNumList(s,1))*sqrt(weights_phi(ph))*nk;
			MJKeMRe(s,ph) = cos(grid_phi(ph)*jkemQuantumNumList(s,2))*sqrt(weights_phi(ph))*nk;
			MJKeMIm(s,ph) = sin(grid_phi(ph)*jkemQuantumNumList(s,2))*sqrt(weights_phi(ph))*nk;
		}
	}
}

void get_JKMbasis(int jkem, int size_theta, int size_phi, matrix &dJKeM, matrix &KJKeMRe, matrix &KJKeMIm, matrix &MJKeMRe, matrix &MJKeMIm, vector &basisJKeMRe, vector &basisJKeMIm) 
{
	double valdJ, valKC, valKS, valMC, valMS, fac1, fac2;
	int i, it, ip, ic, iit, itp, itpc; 

	for (i=0; i<jkem; i++) {

		for (it=0; it<size_theta; it++) {
			iit = it+i*size_theta;
			valdJ=dJKeM(i,it);

			for (ip=0; ip<size_phi; ip++) {
				itp = ip+iit*size_phi;
				valMC=MJKeMRe(i,ip);
				valMS=MJKeMIm(i,ip);

				for (ic=0; ic<size_phi; ic++) {
					itpc = ic+itp*size_phi;

					valKC=KJKeMRe(i,ic);
					valKS=KJKeMIm(i,ic);

					basisJKeMRe(itpc)=valdJ*(valMC*valKC-valMS*valKS);//(c1+i*s1)*(c2+i*s2)
					basisJKeMIm(itpc)=valdJ*(valMC*valKS+valMS*valKC);
				}
			}
		}
	}
}

void check_norm(int jkem, int size_theta, int size_phi, matrix &dJKeM, matrix &KJKeMRe, matrix &KJKeMIm, matrix &MJKeMRe, matrix &MJKeMIm, matrix &jkemQuantumNumList) 
{
	matrix normJKMRe(jkem,jkem);
	matrix normJKMIm(jkem,jkem);

	double lvec1dJ, lvec2dJ, lvecKC1, lvecKS1, lvecKC2, lvecKS2, lvecMC1, lvecMS1, lvecMC2, lvecMS2;
	double rvec1dJ, rvec2dJ, rvecKC1, rvecKS1, rvecKC2, rvecKS2, rvecMC1, rvecMS1, rvecMC2, rvecMS2;
	double fac1, fac2, fac3, fac4;
	int it, ip, ic, i, j;

	for (it=0; it<size_theta; it++) {
		for (ip=0; ip<size_phi; ip++) {
			for (ic=0; ic<size_phi; ic++) {
				for (i=0; i<jkem; i++) {
					lvec1dJ=dJKeM(i,it);

					lvecMC1=MJKeMRe(i,ip);
					lvecMS1=MJKeMIm(i,ip);

					lvecKC1=KJKeMRe(i,ic);
					lvecKS1=KJKeMIm(i,ic);

					fac1=lvecMC1*lvecKC1-lvecMS1*lvecKS1;//(c1+i*s1)*(c2+i*s2)
					fac2=lvecMC1*lvecKS1+lvecMS1*lvecKC1;

					for (int j=0; j<jkem; j++) {
						rvec1dJ=dJKeM(j,it);

						rvecMC1=MJKeMRe(j,ip);
						rvecMS1=MJKeMIm(j,ip);

						rvecKC1=KJKeMRe(j,ic);
						rvecKS1=KJKeMIm(j,ic);

						fac3=rvecMC1*rvecKC1-rvecMS1*rvecKS1;//(c1+i*s1)*(c2+i*s2)
						fac4=rvecMC1*rvecKS1+rvecMS1*rvecKC1;

						normJKMRe(i,j)+=lvec1dJ*rvec1dJ*(fac1*fac3+fac2*fac4);
						normJKMIm(i,j)+=lvec1dJ*rvec1dJ*(fac3*fac2-fac1*fac4);
					}
				}
			}
		}
	}

    ofstream checknorm("norm.txt");
    checknorm<<"# Checking of orthonormality of |JKM> basis"<<endl;
	checknorm<<endl;
    for (i=0; i<jkem; i++)
    {
		for (j=0; j<jkem; j++)
		{
			checknorm<<" "<<jkemQuantumNumList(i,0)<<" "<<jkemQuantumNumList(i,1)<<" "<<jkemQuantumNumList(j,2)<<endl;
			checknorm<<" "<<jkemQuantumNumList(j,0)<<" "<<jkemQuantumNumList(j,1)<<" "<<jkemQuantumNumList(j,2)<<endl;
			checknorm<<" Norm Re: "<<normJKMRe(i,j)<<"    Im: "<<normJKMIm(i,j)<<endl;
			checknorm<<endl;
		}
		checknorm<<endl;
		checknorm<<endl;
	}
	checknorm.close();
}

void check_normJKeM(int jkem, int size_grid, vector &basisJKeMRe, vector &basisJKeMIm, matrix &jkemQuantumNumList) 
{
	matrix normJKMRe(jkem,jkem);
	matrix normJKMIm(jkem,jkem);

	double lvec1dJ, lvec2dJ, lvecKC1, lvecKS1, lvecKC2, lvecKS2, lvecMC1, lvecMS1, lvecMC2, lvecMS2;
	double rvec1dJ, rvec2dJ, rvecKC1, rvecKS1, rvecKC2, rvecKS2, rvecMC1, rvecMS1, rvecMC2, rvecMS2;
	double fac1, fac2, fac3, fac4;
	int ig, i, j;

	for (ig=0; ig<size_grid; ig++) {
		for (i=0; i<jkem; i++) {
			for (j=0; j<jkem; j++) {
				normJKMRe(i,j)+=basisJKeMRe(ig+i*size_grid)*basisJKeMRe(ig+j*size_grid)+basisJKeMIm(ig+i*size_grid)*basisJKeMIm(ig+j*size_grid);
				normJKMIm(i,j)+=basisJKeMRe(ig+i*size_grid)*basisJKeMIm(ig+j*size_grid)-basisJKeMIm(ig+i*size_grid)*basisJKeMRe(ig+j*size_grid);
			}
		}
	}

    ofstream checknorm("norm.txt", ios::app);
    checknorm<<"# Final checking of orthonormality of |JKM> basis in check_normJKeM "<<endl;
	checknorm<<endl;
    for (int i=0; i<jkem; i++)
    {
		for (int j=0; j<jkem; j++)
		{
			checknorm<<" "<<jkemQuantumNumList(i,0)<<" "<<jkemQuantumNumList(i,1)<<" "<<jkemQuantumNumList(j,2)<<endl;
			checknorm<<" "<<jkemQuantumNumList(j,0)<<" "<<jkemQuantumNumList(j,1)<<" "<<jkemQuantumNumList(j,2)<<endl;
			checknorm<<" Norm Re: "<<normJKMRe(i,j)<<"    Im: "<<normJKMIm(i,j)<<endl;
			checknorm<<endl;
		}
		checknorm<<endl;
		checknorm<<endl;
	}
	checknorm.close();
}

void get_pot(int size_theta, int size_phi, vector &grid_theta, vector &grid_phi, double *com1, double *com2, matrix &Vpot)
{
    double theta1, theta2, phi1, phi2, chi1, chi2;
	double Eulang1[3];
	double Eulang2[3];
	double E_2H2O;

    for (int it = 0; it < size_theta; it++)
    {
        theta1 = grid_theta(it);
		Eulang1[1]=theta1;

		for (int ip = 0; ip < size_phi; ip++)
		{
			phi1 = grid_phi(ip);
			Eulang1[0]=phi1;

			for (int ic = 0; ic < size_phi; ic++)
			{
				chi1 = grid_phi(ic);
				Eulang1[2]=chi1;

				for (int jt = 0; jt < size_theta; jt++)
				{
					theta2 = grid_theta(jt);
					Eulang2[1]=theta2;

					for (int jp = 0; jp < size_phi; jp++)
					{
						phi2 = grid_phi(jp);
						Eulang2[0]=phi2;

						for (int jc = 0; jc < size_phi; jc++)
						{
							chi1 = grid_phi(jc);
							Eulang2[2]=chi2;

							caleng_(com1, com2, &E_2H2O, Eulang1, Eulang2);
							Vpot(ic+(ip+it*size_phi)*size_phi, jc+(jp+jt*size_phi)*size_phi) = E_2H2O;
						}
					}
				}
            }
        }
    }
}

cvector Hv(int jkem, int size_grid, vector &HrotKe, matrix &Vpot, vector &basisJKeMRe, vector &basisJKeMIm, cvector &v)
{
	cvector u(jkem*jkem);
	int i1, i1p, i2, i2p;
  
	// oprate with K1
	for (i2=0;i2<jkem;i2++) {
		for (i1=0;i1<jkem;i1++) {
			for (i1p=0;i1p<jkem;i1p++) {
				u(i1*jkem+i2)+=HrotKe(i1p+i1*jkem)*v(i1p*jkem+i2);
			}
		}
	}

	// oprate with K2
	for (i1=0;i1<jkem;i1++) {
		for (i2=0;i2<jkem;i2++) {
			for (i2p=0;i2p<jkem;i2p++) {
				u(i1*jkem+i2)+=HrotKe(i2p+i2*jkem)*v(i1*jkem+i2p);
			}
		}
	}

    // potential term
	int ig1, ig2;
	vector temp1re(jkem*size_grid);
	vector temp1im(jkem*size_grid);
	vector temp2re(size_grid*size_grid);
	vector temp2im(size_grid*size_grid);

	for (i1=0;i1<jkem;i1++) {
		for (ig2=0;ig2<size_grid;ig2++) {
			for (i2=0;i2<jkem;i2++) {
				temp1re(ig2+i1*size_grid)+=(basisJKeMRe(ig2+i2*size_grid)*real(v(i2+i1*jkem))-basisJKeMIm(ig2+i2*size_grid)*imag(v(i2+i1*jkem)));
				temp1im(ig2+i1*size_grid)+=(basisJKeMIm(ig2+i2*size_grid)*real(v(i2+i1*jkem))+basisJKeMRe(ig2+i2*size_grid)*imag(v(i2+i1*jkem)));
			}
		}
	}

	for (ig2=0;ig2<size_grid;ig2++) {
		for (ig1=0;ig1<size_grid;ig1++) {
			for (i1=0;i1<jkem;i1++) {
				temp2re(ig2+ig1*size_grid)+=(basisJKeMRe(ig1+i1*size_grid)*temp1re(ig2+i1*size_grid)-basisJKeMIm(ig1+i1*size_grid)*temp1im(ig2+i1*size_grid));
				temp2im(ig2+ig1*size_grid)+=(basisJKeMRe(ig1+i1*size_grid)*temp1im(ig2+i1*size_grid)+basisJKeMIm(ig1+i1*size_grid)*temp1re(ig2+i1*size_grid));
			}
		}
	}

	for (ig1=0; ig1<size_grid; ig1++) {
		for (ig2=0; ig2<size_grid; ig2++) {
			temp2re(ig2+ig1*size_grid)=Vpot(ig1,ig2)*temp2re(ig2+ig1*size_grid);
			temp2im(ig2+ig1*size_grid)=Vpot(ig1,ig2)*temp2im(ig2+ig1*size_grid);
		}
	}

	for (ig1=0; ig1<size_grid; ig1++) {
		for (i2=0; i2<jkem; i2++) {
			for (ig2=0; ig2<size_grid; ig2++) {
				temp1re(ig1+i2*size_grid)+=(temp2re(ig2+ig1*size_grid)*basisJKeMRe(ig2+i2*size_grid)+temp2im(ig2+ig1*size_grid)*basisJKeMIm(ig2+i2*size_grid));
				temp1im(ig1+i2*size_grid)+=(temp2im(ig2+ig1*size_grid)*basisJKeMRe(ig2+i2*size_grid)-temp2re(ig2+ig1*size_grid)*basisJKeMIm(ig2+i2*size_grid));
			}
		}
	}

	cvector vec(jkem*jkem);
	for (i2=0; i2<jkem; i2++) {
		for (i1=0; i1<jkem; i1++) {
			for (ig1=0; ig1<size_grid; ig1++) {
				vec(i2+i1*jkem)+=complex((temp1re(ig1+i2*size_grid)*basisJKeMRe(ig1+i1*size_grid)+temp1im(ig1+i2*size_grid)*basisJKeMIm(ig1+i1*size_grid)), (temp1im(ig1+i2*size_grid)*basisJKeMRe(ig1+i1*size_grid)-temp1re(ig1+i2*size_grid)*basisJKeMIm(ig1+i1*size_grid)));
			}
		}
	}
	u=u+vec;
	return u;
}
