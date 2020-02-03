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

void lancbis(int niter,vector &eval,vector &evalerr,double elmin, double elmax,int &ngood,const vector& alpha,const vector& beta, const vector& beta2);
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
void check_norm(string fname, int jkem, int size_theta, int size_phi, matrix &dJKeM, matrix &KJKeMRe, matrix &KJKeMIm, matrix &MJKeMRe, matrix &MJKeMIm, matrix &jkemQuantumNumList);
void get_JKMbasis(int jkem, int size_theta, int size_phi, matrix &dJKeM, matrix &KJKeMRe, matrix &KJKeMIm, matrix &MJKeMRe, matrix &MJKeMIm, vector &basisJKeMRe, vector &basisJKeMIm);
void check_normJKeM(string fname, int jkem, int size_grid, vector &basisJKeMRe, vector &basisJKeMIm, matrix &jkemQuantumNumList); 
void get_pot(int size_theta, int size_phi, vector &grid_theta, vector &grid_phi, double *com1, double *com2, matrix &Vpot);
extern "C" void caleng_(double *com1, double *com2, double *E_2H2O, double *Eulang1, double *Eulang2);
cvector Hv(int jkem, int size_grid, vector &HrotKe, matrix &Vpot, vector &basisJKeMRe, vector &basisJKeMIm, cvector &v0);
void printIndex(int jkem);

void lanczosvectors(vector &alpha,vector &beta,vector &beta2,int niter, vector &eval,int ngood,matrix &evtr);
void EVanalysis(vector &grid,int size,int nconv,vector &ARv,double Ri,double Rf, int basistype,int size3d, diagmat &Rfunc,diagmat &rfunc, diagmat &R2func,diagmat &r2func,diagmat &sqrtweight);

EXTERN void FORTRAN(trivec)(double *lalpha,double *lbeta,double *lbeta2, double *wrk1,double *wrk2,int *niter, double *eval,int *ngood,double *evtr,double *mamin);
 
int main(int argc,char **argv) 
{
	const int IO_OUTPUT_WIDTH=3;

    time_t totalstart,totalend,callpotstart,callpotend,diagstart,diagend;
    time (&totalstart);
    char timemsg[100];
	
    double zCOM = atof(argv[1]);
    int jmax = atoi(argv[2]);
	int niter = atoi(argv[3]);

	//The rotational A, B, C constants are indicated by ah2o, bh2o and ch2o, respectively. The unit is cm^-1.
    double ah2o= 27.877;//cm-1
    double bh2o= 14.512;//cm-1
    double ch2o= 9.285; //cm-1
    ah2o=ah2o*CMRECIP2KL;
    bh2o=bh2o*CMRECIP2KL;
    ch2o=ch2o*CMRECIP2KL;

	int size_theta, size_phi;
	if (jmax <= 4) {
		size_theta = 2*jmax+3;
		size_phi   = 2*(2*jmax+1);
	}
	else {
		size_theta = 2*jmax+1;
		size_phi   = 2*jmax+1;
	}

// Generation of names of output file //
	stringstream rprefix, jprefix, iterprefix, thetaprefix, phiprefix;
	rprefix<<fixed<<setprecision(1)<<zCOM;
	jprefix<<jmax;
	iterprefix<<niter;
	phiprefix<<size_phi;
	thetaprefix<<size_theta;
	string fname = "lanc-2-p-H2O-jmax" + jprefix.str()+"-Rpt"+rprefix.str() + "Angstrom-grid-"+thetaprefix.str()+"-"+phiprefix.str()+"-niter"+iterprefix.str()+".txt";
	string fname1="logout-"+fname;
	string fname2="energy-levels-"+fname;
	string fname3="ground-state-energy-"+fname;
//
    ofstream logout(fname1.c_str());

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
	check_normJKeM(fname, jkem, size_grid, basisJKeMRe, basisJKeMIm, jkemQuantumNumList); 

	double com1[3]={0.0, 0.0, 0.0};
	double com2[3]={0.0, 0.0, zCOM};

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

	int size_basis = jkem*jkem;
	cvector v0(size_basis);
	for (int i=0; i<jkem; i++) {
		for (int j=0; j<jkem; j++) {
			v0(j+i*jkem)=complex(2.0*(randomseed->frand()-0.5), 0.0);
		}
	}
	cnormalise(v0);

	cvector v0keep=v0;
	for (int i=0;i<size_basis;i++) r(i)=complex(0.,0.);

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

	double emax=100.0;
	double emin=-2000.0;
	lancbis(niter,eval,evalerr,emin,emax,ngood,alpha,beta,beta2);
	logout<<" ngood = "<<ngood<<endl;
	cout<<"E0 = "<<eval(0)<<endl;
	// lanczos report:
	ofstream lancout2(fname2.c_str());
	ofstream lancout3(fname3.c_str());
	lancout3<<eval(0)<<" "<<evalerr(0)<<endl;
	for (int i=0; i<ngood; i++) lancout2<<eval(i)<<" "<<evalerr(i)<<endl;
	lancout2.flush();
	lancout2.close();
	lancout3.flush();
	lancout3.close();
	logout.close();


	/*
	//computation of eigenvectors begins
	matrix evtr(niter,ngood);
	lanczosvectors(alpha,beta,beta2,niter,eval,ngood,evtr);
    
	vector cumulnorm(ngood);
	for (int j=1;j<=niter;j++) {	
		for (int n=0;n<ngood;n++) {
			double coeff2=pow(evtr(j-1,n),2.);
			cumulnorm(n)+=coeff2;
		}
	}
    
	v0=v0keep;
	cvector ARvL(size_basis*ngood);
	vector testv(size_basis);

	for (int i=0;i<size_basis;i++) r(i)=0.;

	// lanczos vector coefficent matrix
	for (int n=0;n<ngood;n++) cumulnorm(n)=0.;
	for (int j=1;j<=niter;j++) {	
		// tranform vector	
		for (int n=0;n<ngood;n++) {
			double treshold=pow(evtr(j-1,n),2.);
			cumulnorm(n)+=treshold;
			for (int row=0;row<size_basis;row++) {
				//double coeff=evtr(j-1,n)*v0(row);	  
				complex coeff=complex(evtr(j-1,n),0.0)*v0(row);	  
				if (cumulnorm(n) <(1.-1.e-16))
					ARvL(row+size_basis*n)+=coeff;	
			}
		}
		
		u=Hv(jkem, size_grid, HrotKe, Vpot, basisJKeMRe, basisJKeMIm, v0);

		r=r+u;       
		//alpha(j-1)=v0*r;
		alpha(j-1)=real(v0*r);
		//r=r-(alpha(j-1)*v0);
		r=r-(complex(alpha(j-1),0)*v0);
		//beta2(j)=r*r;
		beta2(j)=real(r*r);
		beta(j)=sqrt(beta2(j));
		//r=(1./beta(j))*r; // to get v
		r=complex((1./beta(j)),0)*r; // to get v
		//v0=(-beta(j))*v0; // prepare r check minus sign!!!  
		v0=complex((-beta(j)),0)*v0; // prepare r check minus sign!!!

		u=v0;     // swapping
		v0 =r;
		r=u;	
		
		if (j%100 == 0) logout<<"iteration "<<j<<endl;
    }         
	
	cvector psi0(size_basis);
    // purification step of all eigenvectors
    for (int n=0;n<ngood;n++) {
		for (int row=0;row<size_basis;row++)
			r(row)=ARvL(row+size_basis*n);

        if (n==0) psi0=r;
    }
	
	logout<<psi0*psi0<<endl;
	*/
	exit(111);
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

	double phimin=0.0;
	double phimax=2.0*M_PI;
	double dphi=2.0*M_PI/(double)nsize;
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

	if (jmax%2==0) {
		jkem=int((jkm+jmax+1)/2);
		jkom=jkm-jkem;
	}
	else {
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
	if (chkjkm != jkm2) {
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
	for (int j=0; j<(jmax+1); j++) {
		if (j%2==0) {
			for (int k=-j; k<(j+1); k+=2) {
				for (int m=-j; m<(j+1); m++) {
					jkemQuantumNumList(jtempcounter,0)=j;
					jkemQuantumNumList(jtempcounter,1)=k;
					jkemQuantumNumList(jtempcounter,2)=m;
					jtempcounter++;
				}
			}
		}
		else {
			for (int k=(-j+1); k<j; k+=2) {
				for (int m=-j; m<(j+1); m++) {
					jkemQuantumNumList(jtempcounter,0)=j;
					jkemQuantumNumList(jtempcounter,1)=k;
					jkemQuantumNumList(jtempcounter,2)=m;
					jtempcounter++;
				}
			}
		}
	}
	//logout<<"jtempcounter "<<jtempcounter<<endl;
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

void check_norm(string fname, int jkem, int size_theta, int size_phi, matrix &dJKeM, matrix &KJKeMRe, matrix &KJKeMIm, matrix &MJKeMRe, matrix &MJKeMIm, matrix &jkemQuantumNumList) 
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

	string fname1 = "norm-"+fname;
    ofstream checknorm(fname1.c_str());
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

void check_normJKeM(string fname, int jkem, int size_grid, vector &basisJKeMRe, vector &basisJKeMIm, matrix &jkemQuantumNumList) 
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

	string fname1 = "norm-"+fname;
    ofstream checknorm(fname1.c_str(), ios::app);
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
	double Eulang1[3];
	double Eulang2[3];
	double E_2H2O;

    for (int it=0; it<size_theta; it++) {
		Eulang1[1]=grid_theta(it);
		for (int ip=0; ip<size_phi; ip++) {
			Eulang1[0]=grid_phi(ip);
			for (int ic=0; ic<size_phi; ic++) {
				Eulang1[2]=grid_phi(ic);

				for (int jt=0; jt<size_theta; jt++) {
					Eulang2[1]=grid_theta(jt);
					for (int jp=0; jp<size_phi; jp++) {
						Eulang2[0]=grid_phi(jp);
						for (int jc=0; jc<size_phi; jc++) {
							Eulang2[2]=grid_phi(jc);

							caleng_(com1,com2,&E_2H2O,Eulang1,Eulang2);
							int ii=ic+(ip+it*size_phi)*size_phi;
							int jj=jc+(jp+jt*size_phi)*size_phi;
							Vpot(ii,jj)=E_2H2O;
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
				u(i1*jkem+i2)+=complex(HrotKe(i1p+i1*jkem)*real(v(i1p*jkem+i2)),0.0);
			}
		}
	}

	// oprate with K2
	for (i1=0;i1<jkem;i1++) {
		for (i2=0;i2<jkem;i2++) {
			for (i2p=0;i2p<jkem;i2p++) {
				u(i1*jkem+i2)+=complex(HrotKe(i2p+i2*jkem)*real(v(i1*jkem+i2p)),0.0);
			}
		}
	}

    // potential term
	int ig1, ig2;
	vector temp1re(jkem*size_grid);
	vector temp1im(jkem*size_grid);
	vector temp2re(size_grid*size_grid);
	vector temp2im(size_grid*size_grid);
	vector temp3re(jkem*size_grid);
	vector temp3im(jkem*size_grid);

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
				temp3re(ig1+i2*size_grid)+=(temp2re(ig2+ig1*size_grid)*basisJKeMRe(ig2+i2*size_grid)+temp2im(ig2+ig1*size_grid)*basisJKeMIm(ig2+i2*size_grid));
				temp3im(ig1+i2*size_grid)+=(temp2im(ig2+ig1*size_grid)*basisJKeMRe(ig2+i2*size_grid)-temp2re(ig2+ig1*size_grid)*basisJKeMIm(ig2+i2*size_grid));
			}
		}
	}

	cvector vec(jkem*jkem);
	for (i2=0; i2<jkem; i2++) {
		for (i1=0; i1<jkem; i1++) {
			for (ig1=0; ig1<size_grid; ig1++) {
				vec(i2+i1*jkem)+=complex((temp3re(ig1+i2*size_grid)*basisJKeMRe(ig1+i1*size_grid)+temp3im(ig1+i2*size_grid)*basisJKeMIm(ig1+i1*size_grid)), (temp3im(ig1+i2*size_grid)*basisJKeMRe(ig1+i1*size_grid)-temp3re(ig1+i2*size_grid)*basisJKeMIm(ig1+i1*size_grid)));
			}
		}
	}
	u=u+vec;
	return u;
}

void lanczosvectors(vector &alpha, vector &beta, vector &beta2, int niter, vector &eval, int ngood, matrix &evtr)
{
	// copy stuff
	int i,j,ndis;
	double* lalpha=new double[niter];
	double* lbeta=new double[niter+1];
	double* lbeta2=new double[niter+1];
	double* leval=new double[ngood];
	lbeta[0]=0.;
	lbeta2[0]=0.;
	for (j=1;j<=niter;j++) {
		lalpha[j-1]=alpha(j-1);
		lbeta[j]=beta(j);
		lbeta2[j]=beta2(j);
	}
	for (j=0;j<ngood;j++) leval[j]=eval(j);

	double* wrk1=new double[niter];
	double* wrk2=new double[niter];
	double* levtr=new double[niter*ngood];
	double* mamin=new double[ngood];

	FORTRAN(trivec)(lalpha,lbeta,lbeta2,wrk1,wrk2,&niter,leval,&ngood,levtr,mamin);

	for (i=0;i<niter;i++)
		for (j=0;j<ngood;j++) 
			evtr(i,j)=levtr[i+j*niter];
	return;
}

void EVanalysis(vector &grid,int size,int nconv,vector &ARv,double Ri,double Rf,
				int basistype,int size3d, diagmat &Rfunc,diagmat &rfunc,
				diagmat &R2func,diagmat &r2func,diagmat &sqrtweight)
{
    Rand *randomseed = new Rand(1);
    int i,j,k;
    double r1,r2,r3;
    ofstream evout("ev");
    int column1=0;  
    int column2=1;
    int column3=2;
    int column4=3;
    int column5=4;
    int column6=5;
    
    ofstream drout("dr");
    ofstream Vrout("vr");
    
    int nc;
	
    vector Ravg(nconv);
    vector ravg(nconv);
    vector R2avg(nconv);
    vector r2avg(nconv);
	
    vector countlinear(nconv);
    vector countisoceles(nconv);
    vector countequilateral(nconv);
    vector countscalene(nconv);
    vector countall(nconv);
    
    for (i=0;i<size;i++){      
		vector Dr(nconv);      
		for (j=0;j<size;j++){
			if (basistype == 0) {
				r1=Ri+(grid(i)/fabs(grid(0))+1.)*(Rf-Ri)/2.;
				r2=Ri+(grid(j)/fabs(grid(0))+1.)*(Rf-Ri)/2.;
			}
			else {
				r1=Ri+(grid(i)+1.)*(Rf-Ri)/2.;
				r2=Ri+(grid(j)+1.)*(Rf-Ri)/2.;
			}
			double fuzz=1.+.01*(randomseed->frand()-.5);		  
			vector ARvv(nconv);	
			for (k=0;k<size;k++){
				int row=(i*size+j)*size+k;
				for ( nc=0;nc<nconv;nc++) {
					ARvv(nc)+=pow(ARv(row+size3d*nc),2.);	  	 
					if (i==j && j==k) 
						countequilateral(nc)+=pow(ARv(row+size3d*nc),2.);
					else if (i==0 || j==0 || k==0) 
						countlinear(nc)+=pow(ARv(row+size3d*nc),2.);
					else if (i==j || i==k || j==k) 
						countisoceles(nc)+=pow(ARv(row+size3d*nc),2.);
					else countscalene(nc)+=pow(ARv(row+size3d*nc),2.);
					countall(nc)+=pow(ARv(row+size3d*nc),2.);
					Ravg(nc)+=pow(ARv(row+size3d*nc),2.)*Rfunc((i*size+j)*size+k);
					ravg(nc)+=pow(ARv(row+size3d*nc),2.)*rfunc((i*size+j)*size+k);
					R2avg(nc)+=pow(ARv(row+size3d*nc),2.)*R2func((i*size+j)*size+k);
					r2avg(nc)+=pow(ARv(row+size3d*nc),2.)*r2func((i*size+j)*size+k);
				}	  
				if (basistype == 1 || basistype ==2) {
					r3=Ri+(grid(k)+1.)*(Rf-Ri)/2.;
				}
				if (basistype == 0) {
					r3=Ri+(grid(k)/fabs(grid(0))+1.)*(Rf-Ri)/2.;
				}  
			}
			for ( nc=0;nc<nconv;nc++)
				Dr(nc)+=ARvv(nc);	
			for ( nc=0;nc<nconv;nc++)
				ARvv(nc)*=pow(sqrtweight(i)*sqrtweight(j),2.);
			
			evout<<r1*(1.+.001*(randomseed->frand()-.5))/atob<<" ";
			evout<<r2*(1.+.001*(randomseed->frand()-.5))/atob<<" ";
			for ( nc=0;nc<nconv;nc++) {
				if (basistype == 0)
					evout<<ARvv(nc)/((Rf-Ri)/(grid(size-1)-grid(0))/atob)/((Rf-Ri)/(grid(size-1)-grid(0))/atob)<<" ";
				else
					evout<<ARvv(nc)/(.5*(Rf-Ri)/atob)/(.5*(Rf-Ri)/atob)<<" ";
			}
			evout<<endl;
		}  
		for ( nc=0;nc<nconv;nc++) Dr(nc) *=pow(sqrtweight(i),2.);
		drout<<r1/atob<<" ";
		for ( nc=0;nc<nconv;nc++) {
			if (basistype == 0)
				drout<<Dr(nc)/((Rf-Ri)/(grid(size-1)-grid(0))/atob)<<" ";
			else
				drout<<Dr(nc)/(.5*(Rf-Ri)/atob)<<" ";
		}
		drout<<endl;
    }
	for ( nc=0;nc<nconv;nc++) {
		cout<<"<R>_"<<nc<<" = "<<Ravg(nc)/atob<<" A ; ";
		cout<<"<r>_"<<nc<<" = "<<ravg(nc)/atob<< " A"<<endl;
		cout<<"<R^2>_"<<nc<<" = "<<R2avg(nc)/atob/atob<<" A^2 ; ";
		cout<<"<r^2>_"<<nc<<" = "<<r2avg(nc)/atob/atob<< " A^2"<<endl;
		cout<<"(<R^2>-<R>^2)_"<<nc<<" = "<<(R2avg(nc)-Ravg(nc)*Ravg(nc))/atob/atob<<" A^2 ; ";
		cout<<"(<r>^2-<r^2>)_"<<nc<<" = "<<(r2avg(nc)-ravg(nc)*ravg(nc))/atob/atob<< " A^2"<<endl;
		cout<<"(\% equilateral)_"<<nc<<" = "<<countequilateral(nc)*100.<<endl;
		cout<<"(\% linear)_"<<nc<<" = "<<countlinear(nc)*100.<<endl;
		cout<<"(\% isoceles)_"<<nc<<" = "<<countisoceles(nc)*100.<<endl;
		cout<<"(\% scalene)_"<<nc<<" = "<<countscalene(nc)*100.<<endl;
		cout<<"(\% all)_"<<nc<<" = "<<countall(nc)*100.<<" ";
		cout<<(countscalene(nc)+countisoceles(nc)+countlinear(nc)+countequilateral(nc))*100.<<endl;
	}
	return;
}
