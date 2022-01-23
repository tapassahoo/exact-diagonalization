#include "cmdstuff.h"
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <sstream>
#include <iomanip>
#include "random.h"
#include <complex>
//#include "BF.h"

using namespace std;

static const double AuToAngstrom  = 0.52917720859;
static const double AuToDebye     = 1.0/0.39343;
static const double AuToCmInverse = 219474.63137;
static const double AuToKelvin    = 315777.0;
static const double CMRECIP2KL    = 1.4387672;
static const double pi=3.141592653589793;
static const double small = 10.e-10;

void lancbis(int niter,vector &eval,vector &evalerr,double elmin, double elmax,int &ngood,const vector& alpha,const vector& beta, const vector& beta2);

EXTERNC void gauleg(double x1,double x2,double *x,double *w,int n);
EXTERNC double plgndr(int j,int m,double x);
vector thetagrid(int nsize,vector &weights);
vector phigrid(int nsize,vector &weights);
void get_sizes(int jmax, int *sizes, string fname1);
void get_QuantumNumList_NonLinear_ComplexBasis(int jmax, matrix &jkmList_complex,matrix &jkemList_complex, matrix &jkomList_complex);
double off_diag(int j, int k);
double littleD(int ldJ, int ldmp, int ldm, double ldtheta);
void get_indvbasis(int njkm, int size_theta, vector &grid_theta, vector &weights_theta, int size_phi, vector &grid_phi, vector &weights_phi, matrix &jkemQuantumNumList, matrix &dJKM, matrix &KJKMRe, matrix &KJKMIm, matrix &MJKeMRe, matrix &MJKeMIm);
void get_JKMbasis(int njkm, int size_theta, int size_phi, matrix &dJKM, matrix &KJKMRe, matrix &KJKMIm, matrix &MJKeMRe, matrix &MJKeMIm, cmatrix &wf_complex);
void check_norm_ComplexBasis(string fname1, int njkm, int size_grid, cmatrix &wf_complex, matrix &njkmList_complex, double small);
void get_Hrot(double ah2o, double bh2o, double ch2o, int njkm, matrix &jkemQuantumNumList, matrix &Hrot);

void get_QuantumNumList_NonLinear_RealBasis(int jmax, matrix &jkmList_real,matrix &jkemList_real, matrix &jkomList_real);
void get_wigner_RealBasis(int njkm_J,int njkm_K,int njkm_M,double theta,double wt,double phi,double wp,double chi,double wc, double *theta0, double *thetac, double *thetas);
void get_NonLinear_RealBasis(int jmax,int njkm,int size_theta,int size_phi,vector &grid_theta,vector &weights_theta,vector &grid_phi,vector &weights_phi,matrix &wf_realbasis);
void check_norm_RealBasis(string fname,matrix &wf_realbasis,int njkm,double small);

void get_HrotN2(double ah2o, double bh2o, double ch2o, int njkm, matrix &jkemQuantumNumList, matrix &HrotKee);
void check_norm(string fname, int njkm, int size_theta, int size_phi, matrix &dJKM, matrix &KJKMRe, matrix &KJKMIm, matrix &MJKeMRe, matrix &MJKeMIm, matrix &jkemQuantumNumList);

void get_pot(int size_theta, int size_phi, vector &grid_theta, vector &grid_phi, double *com1, double *com2, matrix &Vpot);
extern "C" void caleng_(double *com1, double *com2, double *E_2H2O, double *Eulang1, double *Eulang2);
vector Hv(int njkm, int size_grid, matrix &Hrot, matrix &Vpot, matrix &wf_real, vector &v0);
void printIndex(int njkm);

void lanczosvectors(vector &alpha,vector &beta,vector &beta2,int niter, vector &eval,int ngood,matrix &evtr);
void EVanalysis(vector &grid,int size,int nconv,vector &ARv,double Ri,double Rf, int basistype,int size3d, diagmat &Rfunc,diagmat &rfunc, diagmat &R2func,diagmat &r2func,diagmat &sqrtweight);

EXTERN void FORTRAN(trivec)(double *lalpha,double *lbeta,double *lbeta2, double *wrk1,double *wrk2,int *niter, double *eval,int *ngood,double *evtr,double *mamin);
 
int main(int argc,char **argv) 
{
	if (argc < 2) 
	{
		cerr << "Wrong number of arguments; Usage: " << argv[0] << " <zCOM> <Jmax> <niter> <Emin> <Emax> <spin_isomer> " << endl;
		exit(0);
	}

	const int IO_OUTPUT_WIDTH=3;

    time_t totalstart,totalend,callpotstart,callpotend,diagstart,diagend;
    time (&totalstart);
    char timemsg[100];
	
	// Reading command-line inputs
    double zCOM = atof(argv[1]);
    int jmax = atoi(argv[2]);
	int niter = atoi(argv[3]);
    double emin = atof(argv[4]);
    double emax = atof(argv[5]);
    string spin_isomer = argv[6];

	// The rotational A, B, C constants are indicated by ah2o, bh2o and ch2o, respectively. The unit is cm^-1.
    double ah2o= 27.877;//cm-1
    double bh2o= 14.512;//cm-1
    double ch2o= 9.285; //cm-1

	// All the units are transformed to inverse kelvin unit.
    ah2o=ah2o*CMRECIP2KL;
    bh2o=bh2o*CMRECIP2KL;
    ch2o=ch2o*CMRECIP2KL;

	int size_theta, size_phi;
	if (jmax <= 8) {
		size_theta = 20;//2*jmax+12;
		size_phi   = 2*(2*jmax+2);
	}
	else {
		size_theta = 2*jmax+1;
		size_phi   = 2*jmax+10;
	}

	// Generation of names of output file 
	stringstream rprefix, jprefix, iterprefix, thetaprefix, phiprefix;
	rprefix<<fixed<<setprecision(2)<<zCOM;
	jprefix<<jmax;
	iterprefix<<niter;
	phiprefix<<size_phi;
	thetaprefix<<size_theta;
	string fname = "lanc-2-p-H2O-jmax" + jprefix.str()+"-Rpt"+rprefix.str() + "Angstrom-grid-"+thetaprefix.str()+"-"+phiprefix.str()+"-niter"+iterprefix.str()+".txt";
	string fname1="logout-"+fname;
	string fname2="energy-levels-"+fname;
	string fname3="ground-state-energy-"+fname;
	string fname4="ground-state-entropies-"+fname;
	string fname5="ground-state-theta-distribution-"+fname;

	// File opening for printing some system informations
    ofstream logout(fname1.c_str(),ios::app);
	if (!logout)
	{
		cout << "Error opening file for logout.." << endl ;
		exit(1);
	}

	// Grid points and the corresponding weights in DVR 
    vector weights_theta(size_theta);
    vector weights_phi(size_phi); 
    vector grid_theta = thetagrid(size_theta, weights_theta);
    vector grid_phi   = phigrid(size_phi, weights_phi);

    // Evaluation of number of basis functions due to coupling to nuchear spin
    int sizes[3];
    get_sizes(jmax, sizes, fname1);
	int jkm=sizes[0];
	int jkem=sizes[1];
	int jkom=sizes[2];

	// Estimations of quantum numbers associated with COMPLEX Wigner basis functions for various nuclear spin coupling
	matrix jkmList_complex(jkm,3);
	matrix jkemList_complex(jkem,3);
	matrix jkomList_complex(jkom,3);
	get_QuantumNumList_NonLinear_ComplexBasis(jmax, jkmList_complex,jkemList_complex, jkomList_complex);

	int njkm;

	if (spin_isomer == "spinless") 
	{
		njkm = jkm;
	}
	else if (spin_isomer == "para")
	{
		njkm = jkem;
	}
	else if (spin_isomer == "ortho")
	{
		njkm = jkom;
	}

	matrix njkmList_complex(njkm,3);
	if (spin_isomer == "spinless") 
	{
		njkmList_complex = jkmList_complex;
	}
	else if (spin_isomer == "para")
	{
		njkmList_complex = jkemList_complex;
	}
	else if (spin_isomer == "ortho")
	{
		njkmList_complex = jkomList_complex;
	}

	// Calling functions for individual and total wavefunctions 
	matrix dJKM(njkm,size_theta);
	matrix KJKMRe(njkm,size_phi);
	matrix KJKMIm(njkm,size_phi);
	matrix MJKMRe(njkm,size_phi);
	matrix MJKMIm(njkm,size_phi);

	// Individual wavefunctions as functions of theta, phi and chi
	get_indvbasis(njkm, size_theta, grid_theta, weights_theta, size_phi, grid_phi, weights_phi, njkmList_complex, dJKM, KJKMRe, KJKMIm, MJKMRe, MJKMIm);
	//check_norm(njkm, size_theta, size_phi, dJKM, KJKMRe, KJKMIm, MJKeMRe, MJKeMIm, jkemQuantumNumList);
	
	// Toal system wavefunction (Complex Wigner) - |JKM>
	int size_grid = size_theta*size_phi*size_phi;
	matrix basisJKMRe(njkm,size_grid);
	matrix basisJKMIm(njkm,size_grid);
	cmatrix wf_complex(njkm,size_grid);
	get_JKMbasis(njkm, size_theta, size_phi, dJKM, KJKMRe, KJKMIm, MJKMRe, MJKMIm, wf_complex);

	// Calling a function to test <JKM|J'K'M'> = delta_(JJ'KK'MM')
	check_norm_ComplexBasis(fname1, njkm, size_grid, wf_complex, njkmList_complex, small); 

	// Calling of rotational kinetic energy operator - <i|K|j>
	matrix Hrot(njkm,njkm);
	get_Hrot(ah2o, bh2o, ch2o, njkm, njkmList_complex, Hrot);
	
	// Set parameters for potential function
	double com1[3]={0.0, 0.0, 0.0};
	double com2[3]={0.0, 0.0, zCOM};

	matrix Vpot(size_grid, size_grid);
	get_pot(size_theta, size_phi, grid_theta, grid_phi, com1, com2, Vpot);

	// Estimations of quantum numbers associated with REAL Wigner basis functions for various nuclear spin coupling
	matrix jkmList_real(jkm,3);
	matrix jkemList_real(jkem,3);
	matrix jkomList_real(jkom,3);
	get_QuantumNumList_NonLinear_RealBasis(jmax, jkmList_real,jkemList_real, jkomList_real);

	matrix njkmList_real(njkm,3);
	if (spin_isomer == "spinless") 
	{
		njkmList_real = jkmList_real;
	}
	else if (spin_isomer == "para")
	{
		njkmList_real = jkemList_real;
	}
	else if (spin_isomer == "ortho")
	{
		njkmList_real = jkomList_real;
	}

	// Real wigner basis implemented here
	matrix wf_real(njkm,size_grid);
	get_NonLinear_RealBasis(jmax, njkm, size_theta, size_phi,grid_theta,weights_theta,grid_phi,weights_phi,wf_real);
	check_norm_RealBasis(fname1,wf_real,njkm,small);

	// Evaluation of umat = <x_j|t_i>
	cmatrix umat(njkm,njkm);
	umat = wf_complex*transpose(complexm(wf_real));	

	logout<<""<<endl;
	logout<<"#*******************************************************"<<endl;
	logout<<""<<endl;
	logout<<"# Printing of diagoanl U'U(i,j)=delta_<ij> for testing the properties of Unitary matix "<<endl;
	logout<<""<<endl;
	logout<<"#*******************************************************"<<endl;

	cmatrix uumat=umat*transpose(umat);
	for (int i=0; i<njkm; i++)
	{
		for (int j=0; j<njkm; j++)
		{
			if (abs(uumat(i,j))>small)
			{
				logout<<setw(IO_OUTPUT_WIDTH)<<i<<setw(IO_OUTPUT_WIDTH)<<j<<"   "<<uumat(i,j)<<endl;
			}
		}
	}

	logout<<""<<endl;
	logout<<"#*******************************************************"<<endl;
	logout<<""<<endl;
	cmatrix Hrot1 = transpose(umat)*complexm(Hrot)*umat;
	for (int i=0; i<njkm; i++)
	{
		for (int j=0; j<njkm; j++)
		{
			if (abs(imag(Hrot1(i,j)))>small)
			{
            	cout<<"Warning, non-real for rot matrix in real basis"<<endl;
				cout<<i<<"      " <<j<<"       "<<Hrot1(i,j)<<endl;
            	exit(111);
			}
		}
	}
	Hrot = realm(Hrot1);

	// Checking hermiticity of <i|Hrot|j>

	for (int i=0; i<njkm; i++)
	{
		for (int j=0; j<njkm; j++)
		{
			if (abs(Hrot(i,j)-Hrot(j,i))>small)
			{
				logout<<i<<"Warning non-hermit Hrot"<<IO_OUTPUT_WIDTH<<i<<IO_OUTPUT_WIDTH<<"    "<<Hrot(i,j)<<"    "<<Hrot(j,i)<<endl;
			}
		}
	}

	//Lanczos
	Rand *randomseed = new Rand(1);

	// PI lanczos
	int ngood;

	vector r(njkm*njkm);
	vector evalerr(niter);
	vector eval(niter);
	vector alpha(niter);
	vector beta(niter+1);
	vector beta2(niter+1);

	logout<<""<<endl;
	logout<<"#*******************************************************"<<endl;
	logout<<""<<endl;
	int size_basis = njkm*njkm;
	logout<<"size of basis = "<<size_basis<<endl;
	logout<<"size of grids = "<<size_grid<<endl;
	logout<<"#*******************************************************"<<endl;
	logout<<""<<endl;

	vector v0(size_basis);
	for (int i=0; i<njkm; i++) {
		for (int j=0; j<njkm; j++) {
			v0(j+i*njkm)=2.0*(randomseed->frand()-0.5);
		}
	}
	normalise(v0);

	vector v0keep=v0;
	for (int i=0;i<size_basis;i++) r(i)=0.0;

	vector u=v0;
	logout<<"start iterations"<<endl;
	for (int i=1;i<=niter;i++) {

		u=Hv(njkm, size_grid, Hrot, Vpot, wf_real, v0);

		r=r+u;

		alpha(i-1)=v0*r;
		r=r-(alpha(i-1)*v0);

		beta2(i)=r*r;
		beta(i)=sqrt(beta2(i));
		r=(1./beta(i))*r; // to get v
		v0=(-beta(i))*v0; // prepare r check minus sign!!!

		u=v0;     // swapping
		v0=r;
		r=u;
		if (i%1 == 0) logout<<"iteration "<<i<<endl;
	}                  

	//double emax=100.0;
	//double emin=-3000.0;
	lancbis(niter,eval,evalerr,emin,emax,ngood,alpha,beta,beta2);
	logout<<" ngood = "<<ngood<<endl;
	cout<<"E0 = "<<eval(0)<<endl;

	// lanczos report:
	ofstream lancout2(fname2.c_str());

	if( ! lancout2 )	{
		cout << "Error opening file for output:" <<fname2<< endl ;
		return -1 ;
	}

	ofstream lancout3(fname3.c_str());

	if( ! lancout3 )	{
		cout << "Error opening file for output" << fname3<<endl ;
		return -1 ;
	}

	lancout3<<eval(0)<<" "<<evalerr(0)<<endl;
	for (int i=0; i<ngood; i++) lancout2<<eval(i)<<" "<<evalerr(i)<<endl;
	lancout2.flush();
	lancout2.close();
	lancout3.flush();
	lancout3.close();

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
	vector ARvL(size_basis*ngood);
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
				double coeff=evtr(j-1,n)*v0(row);	  
				if (cumulnorm(n) <(1.-1.e-16))
					ARvL(row+size_basis*n)+=coeff;	
			}
		}
		
		u=Hv(njkm, size_grid, Hrot, Vpot, wf_real, v0);

		r=r+u;       
		alpha(j-1)=v0*r;
		r=r-(alpha(j-1)*v0);
		beta2(j)=r*r;
		beta(j)=sqrt(beta2(j));
		r=(1./beta(j))*r; // to get v
		v0=(-beta(j))*v0; // prepare r check minus sign!!!  

		u=v0;     // swapping
		v0 =r;
		r=u;	
		
		if (j%1 == 0) logout<<"iteration "<<j<<endl;
    }         
	
	vector psi0(size_basis);
    // purification step of all eigenvectors
    for (int n=0;n<ngood;n++) {
		for (int row=0;row<size_basis;row++)
			r(row)=ARvL(row+size_basis*n);

        if (n==0) psi0=r;
    }
	
	logout<<psi0*psi0<<endl;

	matrix psi0_mat(njkm,njkm);
	for (int i=0; i<njkm; i++) {
		for (int j=0; j<njkm; j++) {
			psi0_mat(i,j)=psi0(j+i*njkm);
		}
	}
	vector svd_alpha=resvd(psi0_mat);
	double sums2=0.0;
	double sumsvn=0.0;
    for (int i = 0; i<njkm; i++) {
		sums2+=pow(svd_alpha(i),4);
		sumsvn+=svd_alpha(i)*svd_alpha(i)*log(svd_alpha(i)*svd_alpha(i));
    }
	double S_2 = -log(sums2);
	double S_vN = -sumsvn;
	
	ofstream lancout4(fname4.c_str());
	if ( !lancout4 ){
		cout << "Error opening file for output:" <<fname4<< endl ;
		return -1 ;
	}

    lancout4<<S_vN<<endl;
    lancout4<<S_2<<endl;
	lancout4.close();

	/* computation of reduced density matrix */
	cmatrix reduced_density(njkm,jmax+1);
	for (int i=0; i<njkm; i++) {
		for (int ip=0; ip<njkm; ip++) {
			if ((njkmList_complex(i,1)==njkmList_complex(ip,1)) and (njkmList_complex(i,2)==njkmList_complex(ip,2))) {
				complex sum2=(0.0,0.0);
				for (int j=0; j<njkm; j++) {
					sum2+=conj(psi0(j+i*njkm))*psi0(j+ip*njkm);
				}
				reduced_density(i,njkmList_complex(ip,0))=sum2;
			}
		}
	}

    ofstream densout(fname5.c_str());
	complex sum3=(0.0,0.0);
	for (int th=0; th<size_theta; th++) {
		complex sum1=(0.0,0.0);
		for (int i=0; i<njkm; i++) {
			for (int ip=0; ip<njkm; ip++) {
				if ((njkmList_complex(i,1)==njkmList_complex(ip,1)) and (njkmList_complex(i,2)==njkmList_complex(ip,2))) {
					sum1+=4.0*M_PI*M_PI*reduced_density(i,njkmList_complex(ip,0))*dJKM(i,th)*dJKM(ip,th);
				}
			}
		}
		densout<<cos(grid_theta(th))<<"   "<<real(sum1)/weights_theta(th)<<endl;
		sum3+=sum1;
	}
	densout.close();
 	
	logout<<"Normalization: reduced density matrix = "<<sum3<<endl;

	logout.close();
	exit(1);
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
		grid(i)=acos(x[i]); // grids of thete only not cos(theta)
		weights(i)=w[i];    // in this case weights_theta=weights since the argument uses a reference operator
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

void get_sizes(int jmax, int *sizes, string fname1)
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

    ofstream logout(fname1.c_str());
	if (chkjkm != jkm2) {
		logout<<"Wrong index estimation ..."<<endl;
		exit(1);
	}
    
	logout<<""<<endl;
	logout<<"#************************************************"<<endl;
	logout<<"# Number of basis functions calculations ...."<<endl;
	logout<<"# "<<endl;
	logout<<"# Number of |JKM> basis = "<<jkm<<endl;
	logout<<"# "<<endl;
	logout<<"# Number of even K in |JKM> = "<<jkem<<endl;
	logout<<"# Number of odd  K in |JKM> = "<<jkom<<endl;
	logout<<"# "<<endl;
	logout<<"# Number of even K1, even K2 in the |J1K1M1,J2K2M2> = "<<jkeem<<endl;
	logout<<"# Number of even K1, odd  K2 in the |J1K1M1,J2K2M2> = "<<jkeom<<endl;
	logout<<"# Number of odd  K1, even K2 in the |J1K1M1,J2K2M2> = "<<jkoem<<endl;
	logout<<"# Number of odd  K1, odd  K2 in the |J1K1M1,J2K2M2> = "<<jkoom<<endl;
	logout<<"# "<<endl;
	logout<<"# Number of |J1K1M1;J2K2M2> basis = Number of ChkJKM"<<endl;
	logout<<"# Number of |J1K1M1;J2K2M2> basis = "<<jkm2<<endl;
	logout<<"# Number of ChkJKM = "<<chkjkm<<endl;
	logout<<"#************************************************"<<endl;
	logout<<""<<endl;
	logout.close();
}

void get_QuantumNumList_NonLinear_ComplexBasis(int jmax, matrix &jkmList_complex,matrix &jkemList_complex, matrix &jkomList_complex)
{
	/*
    Lists of (J,K,M) quantum number indices computed for nuclear spin isomers

    Para isomer is obtained by summing over even K,
    Ortho isomer is obtained by summing over odd K,
    spinless is computed by summing over all K values.
	*/

	// Spinless system: Sum over All K
	int jtempcounter=0;
	for (int j=0; j<(jmax+1); j++) {
		for (int k=-j; k<(j+1); k++) {
			for (int m=-j; m<(j+1); m++) {
				jkmList_complex(jtempcounter,0)=j;
				jkmList_complex(jtempcounter,1)=k;
				jkmList_complex(jtempcounter,2)=m;
				jtempcounter++;
			}
		}
	}

	// Para spin isomer: Sum over even K
	jtempcounter=0;
	for (int j=0; j<(jmax+1); j++) {
		if (j%2==0) {
			for (int k=-j; k<(j+1); k+=2) {
				for (int m=-j; m<(j+1); m++) {
					jkemList_complex(jtempcounter,0)=j;
					jkemList_complex(jtempcounter,1)=k;
					jkemList_complex(jtempcounter,2)=m;
					jtempcounter++;
				}
			}
		}
		else {
			for (int k=(-j+1); k<j; k+=2) {
				for (int m=-j; m<(j+1); m++) {
					jkemList_complex(jtempcounter,0)=j;
					jkemList_complex(jtempcounter,1)=k;
					jkemList_complex(jtempcounter,2)=m;
					jtempcounter++;
				}
			}
		}
	}
	//logout<<"jtempcounter "<<jtempcounter<<endl;
	
    // Ortho spin isomer: Sum over odd K
	jtempcounter=0;
	for (int j=0; j<(jmax+1); j++)
	{
		if (j%2==0)
		{
			for (int k=(-j+1); k<j; k+=2)
			{
				for (int m=-j; m<(j+1); m++)
				{
					jkomList_complex(jtempcounter,0)=j;
					jkomList_complex(jtempcounter,1)=k;
					jkomList_complex(jtempcounter,2)=m;
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
					jkomList_complex(jtempcounter,0)=j;
					jkomList_complex(jtempcounter,1)=k;
					jkomList_complex(jtempcounter,2)=m;
					jtempcounter++;
				}
			}
		}
	}
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

void get_Hrot(double ah2o, double bh2o, double ch2o, int njkm, matrix &jkemQuantumNumList, matrix &Hrot)
{
    //Computation of Hrot (Asymmetric Top Hamiltonian in Symmetric Top Basis)
	int jkm1, jkmp1;
	for (jkm1=0; jkm1<njkm; jkm1++) {
		for (jkmp1=0; jkmp1<njkm; jkmp1++) {

			if ((jkemQuantumNumList(jkm1,0)==jkemQuantumNumList(jkmp1,0)) && (jkemQuantumNumList(jkm1,2)==jkemQuantumNumList(jkmp1,2)))
			{
				if (jkemQuantumNumList(jkm1,1)==(jkemQuantumNumList(jkmp1,1)-2))
				{
					Hrot(jkm1,jkmp1) += 0.25*(ah2o-ch2o)*off_diag(jkemQuantumNumList(jkm1,0),jkemQuantumNumList(jkm1,1))*off_diag(jkemQuantumNumList(jkm1,0),jkemQuantumNumList(jkm1,1)+1);
				}
				else if (jkemQuantumNumList(jkm1,1)==(jkemQuantumNumList(jkmp1,1)+2))
				{
					Hrot(jkm1,jkmp1) += 0.25*(ah2o-ch2o)*off_diag(jkemQuantumNumList(jkm1,0),jkemQuantumNumList(jkm1,1)-1)*off_diag(jkemQuantumNumList(jkm1,0),jkemQuantumNumList(jkm1,1)-2);
				}
				else if (jkemQuantumNumList(jkm1,1)==(jkemQuantumNumList(jkmp1,1)))
				{
					Hrot(jkmp1,jkmp1) += (0.5*(ah2o + ch2o)*(jkemQuantumNumList(jkm1,0)*(jkemQuantumNumList(jkm1,0)+1)) + (bh2o - 0.5*(ah2o+ch2o)) * pow(jkemQuantumNumList(jkm1,1),2));
				}
			}
		}
	}
}

void get_HrotN2(double ah2o, double bh2o, double ch2o, int njkm, matrix &jkemQuantumNumList, matrix &HrotKee)
{
    //Computation of Hrot (Asymmetric Top Hamiltonian in Symmetric Top Basis)
	long int jkm12 = 0;
	for (int jkm1=0; jkm1<njkm; jkm1++)
	{
		for (int jkm2=0; jkm2<njkm; jkm2++)
		{
			long int jkmp12 = 0;
			for (int jkmp1=0; jkmp1<njkm; jkmp1++)
			{
				for (int jkmp2=0; jkmp2<njkm; jkmp2++)
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

void get_indvbasis(int njkm, int size_theta, vector &grid_theta, vector &weights_theta, int size_phi, vector &grid_phi, vector &weights_phi, matrix &njkmList_complex, matrix &dJKM, matrix &KJKMRe, matrix &KJKMIm, matrix &MJKMRe, matrix &MJKMIm)
{
	/*
	Constructions of individual wavefunctions, dJKM, KJKMRe, KJKMIm, MJKMRe, MJKMIm 
	See page 86 and 105 in the book ``Anglar Momentum'' written by R. N. Zare.
	*/
	  
	double nk=1.0;
	for (int s=0; s<njkm; s++)
	{
		for (int th=0; th<size_theta; th++)
		{
			dJKM(s,th) = sqrt((2.0*njkmList_complex(s,0)+1.0)/(8.0*pow(pi,2)))*littleD(njkmList_complex(s,0),njkmList_complex(s,2),njkmList_complex(s,1),grid_theta(th))*sqrt(weights_theta(th));
		}

		for (int ph=0; ph<size_phi; ph++)
		{
			KJKMRe(s,ph) = cos(grid_phi(ph)*njkmList_complex(s,1))*sqrt(weights_phi(ph))*nk;
			KJKMIm(s,ph) = sin(grid_phi(ph)*njkmList_complex(s,1))*sqrt(weights_phi(ph))*nk;
			MJKMRe(s,ph) = cos(grid_phi(ph)*njkmList_complex(s,2))*sqrt(weights_phi(ph))*nk;
			MJKMIm(s,ph) = sin(grid_phi(ph)*njkmList_complex(s,2))*sqrt(weights_phi(ph))*nk;
		}
	}
}

void get_JKMbasis(int njkm, int size_theta, int size_phi, matrix &dJKM, matrix &KJKMRe, matrix &KJKMIm, matrix &MJKMRe, matrix &MJKMIm, cmatrix &wf_complex) 
{
	int size_grid = size_theta*size_phi*size_phi;
	matrix basisJKMRe(njkm,size_grid);
	matrix basisJKMIm(njkm,size_grid);

	double valdJ, valKC, valKS, valMC, valMS, fac1, fac2;
	int i, it, ip, ic, itp, itpc; 

	for (i=0; i<njkm; i++) {

		for (it=0; it<size_theta; it++) {
			valdJ=dJKM(i,it);

			for (ip=0; ip<size_phi; ip++) {
				itp = ip+it*size_phi;
				valMC=MJKMRe(i,ip);
				valMS=MJKMIm(i,ip);

				for (ic=0; ic<size_phi; ic++) {
					itpc = ic+itp*size_phi;

					valKC=KJKMRe(i,ic);
					valKS=KJKMIm(i,ic);

					basisJKMRe(i,itpc)=valdJ*(valMC*valKC-valMS*valKS);//(c1+i*s1)*(c2+i*s2)
					basisJKMIm(i,itpc)=valdJ*(valMC*valKS+valMS*valKC);
				}
			}
		}
	}
	wf_complex = complexm(basisJKMRe,basisJKMIm);
}

void check_norm_indiv_ComplexBasis(string fname, int njkm, int size_theta, int size_phi, matrix &dJKM, matrix &KJKMRe, matrix &KJKMIm, matrix &MJKMRe, matrix &MJKMIm, matrix &njkmList_complex) 
{
	matrix normJKMRe(njkm,njkm);
	matrix normJKMIm(njkm,njkm);

	double lvec1dJ, lvec2dJ, lvecKC1, lvecKS1, lvecKC2, lvecKS2, lvecMC1, lvecMS1, lvecMC2, lvecMS2;
	double rvec1dJ, rvec2dJ, rvecKC1, rvecKS1, rvecKC2, rvecKS2, rvecMC1, rvecMS1, rvecMC2, rvecMS2;
	double fac1, fac2, fac3, fac4;
	int it, ip, ic, i, j;

	for (it=0; it<size_theta; it++) {
		for (ip=0; ip<size_phi; ip++) {
			for (ic=0; ic<size_phi; ic++) {
				for (i=0; i<njkm; i++) {
					lvec1dJ=dJKM(i,it);

					lvecMC1=MJKMRe(i,ip);
					lvecMS1=MJKMIm(i,ip);

					lvecKC1=KJKMRe(i,ic);
					lvecKS1=KJKMIm(i,ic);

					fac1=lvecMC1*lvecKC1-lvecMS1*lvecKS1;//(c1+i*s1)*(c2+i*s2)
					fac2=lvecMC1*lvecKS1+lvecMS1*lvecKC1;

					for (int j=0; j<njkm; j++) {
						rvec1dJ=dJKM(j,it);

						rvecMC1=MJKMRe(j,ip);
						rvecMS1=MJKMIm(j,ip);

						rvecKC1=KJKMRe(j,ic);
						rvecKS1=KJKMIm(j,ic);

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
    checknorm<<"# Checking of orthonormality ---> <JKM|JKM> = delta_JJ'KK'MM' "<<endl;
	checknorm<<endl;
    for (i=0; i<njkm; i++)
    {
		for (j=0; j<njkm; j++)
		{
			checknorm<<" "<<njkmList_complex(i,0)<<" "<<njkmList_complex(i,1)<<" "<<njkmList_complex(j,2)<<endl;
			checknorm<<" "<<njkmList_complex(j,0)<<" "<<njkmList_complex(j,1)<<" "<<njkmList_complex(j,2)<<endl;
			checknorm<<" Norm Re: "<<normJKMRe(i,j)<<"    Im: "<<normJKMIm(i,j)<<endl;
			checknorm<<endl;
		}
		checknorm<<endl;
		checknorm<<endl;
	}
	checknorm.close();
}

void check_norm_ComplexBasis(string fname1, int njkm, int size_grid, cmatrix &wf_complex, matrix &njkmList_complex, double small) 
{
	cmatrix normJKM = wf_complex*transpose(wf_complex);

    ofstream checknorm(fname1.c_str(), ios::app);
	checknorm<<"#************************************************"<<endl;
    checknorm<<""<<endl;
    checknorm<<"# Final checking of orthonormality of complex |JKM> basis functions --> <JKM|JKM> = delta_JJ'KK'MM' "<<endl;
    checknorm<<""<<endl;

    for (int i=0; i<njkm; i++)
    {
		for (int j=0; j<njkm; j++)
		{
			if (abs(normJKM(i,j)) > small)
			{
				checknorm<<" "<<njkmList_complex(i,0)<<" "<<njkmList_complex(i,1)<<" "<<njkmList_complex(j,2)<<endl;
				checknorm<<" "<<njkmList_complex(j,0)<<" "<<njkmList_complex(j,1)<<" "<<njkmList_complex(j,2)<<endl;
				checknorm<<" Norm Re: "<<realm(normJKM)(i,j)<<"    Im: "<<imagm(normJKM)(i,j)<<endl;
			}
		}
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

vector Hv(int njkm, int size_grid, matrix &Hrot, matrix &Vpot, matrix &wf_real, vector &v)
{
	vector u(njkm*njkm);
  
	// oprate with K1
#pragma omp parallel for 
	for (int i2=0;i2<njkm;i2++) {
		for (int i1=0;i1<njkm;i1++) {
			for (int i1p=0;i1p<njkm;i1p++) {
				u(i1*njkm+i2)+=Hrot(i1,i1p)*v(i1p*njkm+i2);
			}
		}
	}

	// oprate with K2
#pragma omp parallel for 
	for (int i1=0;i1<njkm;i1++) {
		for (int i2=0;i2<njkm;i2++) {
			for (int i2p=0;i2p<njkm;i2p++) {
				u(i1*njkm+i2)+=Hrot(i2,i2p)*v(i1*njkm+i2p);
			}
		}
	}

    // potential term
	vector temp1(njkm*size_grid);
	vector temp2(size_grid*size_grid);

#pragma omp parallel for 
	for (int i1=0;i1<njkm;i1++) {
		for (int ig2=0;ig2<size_grid;ig2++) {
			for (int i2=0;i2<njkm;i2++) {
				temp1(ig2+i1*size_grid)+=wf_real(i2,ig2)*v(i2+i1*njkm);
			}
		}
	}

#pragma omp parallel for 
	for (int ig2=0;ig2<size_grid;ig2++) {
		for (int ig1=0;ig1<size_grid;ig1++) {
			for (int i1=0;i1<njkm;i1++) {
				temp2(ig2+ig1*size_grid)+=wf_real(i1,ig1)*temp1(ig2+i1*size_grid);
			}
			temp2(ig2+ig1*size_grid)=Vpot(ig1,ig2)*temp2(ig2+ig1*size_grid);
		}
	}

#pragma omp parallel for 
	for (int i2=0; i2<njkm; i2++) {
		for (int ig1=0; ig1<size_grid; ig1++) {
			temp1(ig1+i2*size_grid)=0.0;
			for (int ig2=0; ig2<size_grid; ig2++) {
				temp1(ig1+i2*size_grid)+=temp2(ig2+ig1*size_grid)*wf_real(i2,ig2);
			}
		}
	}

	vector vec(njkm*njkm);
#pragma omp parallel for 
	for (int i2=0; i2<njkm; i2++) {
		for (int i1=0; i1<njkm; i1++) {
			for (int ig1=0; ig1<size_grid; ig1++) {
				vec(i2+i1*njkm)+=temp1(ig1+i2*size_grid)*wf_real(i1,ig1);
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

void get_wigner_RealBasis(int njkm_J,int njkm_K,int njkm_M,double theta,double wt,double phi,double wp,double chi,double wc, double *theta0, double *thetac, double *thetas)
{

	/* See ``Appendix: Real basis of non-linear rotor'' in Rep. Prog. Phys. vol. 77 page- 046601 (2014). */

	*theta0 = sqrt((2.*njkm_J+1)/(8.*M_PI*M_PI))*littleD(njkm_J,0,0,theta)*sqrt(wt*wp*wc);
	double dd = sqrt((2.*njkm_J+1)/(4.*M_PI*M_PI))*littleD(njkm_J,njkm_M,njkm_K,theta)*sqrt(wt);
	*thetac = dd*cos(phi*njkm_M+chi*njkm_K)*sqrt(wp)*sqrt(wc);
	*thetas = dd*sin(phi*njkm_M+chi*njkm_K)*sqrt(wp)*sqrt(wc);
}

void get_NonLinear_RealBasis(int jmax,int njkm,int size_theta,int size_phi,vector &grid_theta,vector &weights_theta,vector &grid_phi,vector &weights_phi,matrix &basisf)
{

	/* See ``Appendix: Real basis of non-linear rotor'' in Rep. Prog. Phys. vol. 77 page- 046601 (2014).  */

	int th, ph, ch, itp, itpc, ib, j, k, m;
	double theta, wt, phi, wp, chi, wc;
	double theta0, thetac, thetas;

	for (th=0; th<size_theta; th++)
	{
		theta=grid_theta(th);
		wt=weights_theta(th);

		for (ph=0; ph<size_phi; ph++)
		{
			phi=grid_phi(ph);
			wp=weights_phi(ph);
			itp=ph+th*size_phi;

			for (ch=0; ch<size_phi; ch++)
			{
				chi=grid_phi(ch);
				wc=weights_phi(ch);
				itpc=ch+itp*size_phi;

				ib=0;
				for (j=0; j<(jmax+1); j++)
				{
					k=0;

					m=0;
					theta0 = 0.0; thetac = 0.0; thetas = 0.0;
					get_wigner_RealBasis(j,k,m,theta,wt,phi,wp,chi,wc,&theta0,&thetac,&thetas);
					basisf(ib,itpc)=theta0;
					ib=ib+1;

					for (m=1; m<(j+1); m++)
					{
						theta0 = 0.0; thetac = 0.0; thetas = 0.0;
						get_wigner_RealBasis(j,k,m,theta,wt,phi,wp,chi,wc,&theta0,&thetac,&thetas);
						basisf(ib,itpc)=thetac;
						ib=ib+1;
						basisf(ib,itpc)=thetas;
						ib=ib+1;
					}
						
					for (k=1; k<(j+1); k++)
					{
						for (m=-j; m<(j+1); m++)
						{
							theta0 = 0.0; thetac = 0.0; thetas = 0.0;
							get_wigner_RealBasis(j,k,m,theta,wt,phi,wp,chi,wc,&theta0,&thetac,&thetas);
							basisf(ib,itpc)=thetac;
							ib=ib+1;
							basisf(ib,itpc)=thetas;
							ib=ib+1;
						}
					}
				}
			}
		}
	}	
}

void check_norm_RealBasis(string fname1,matrix &wf_realbasis,int njkm,double small)
{
	// It checks if the real wigner basis functions are normalized? 

	matrix normJKM = wf_realbasis*transpose(wf_realbasis);

    ofstream checknorm(fname1.c_str(),ios::app);
	checknorm <<""<< endl;
	checknorm <<"#*******************************************************"<<endl;
	checknorm <<""<< endl;
	checknorm <<"# Normalization conditions for Real Wigner basis set."<<endl;
	checknorm <<""<< endl;
	checknorm <<"#*******************************************************"<<endl;
	checknorm <<""<< endl;

	for (int s1=0; s1<njkm; s1++)
	{
		for (int s2=0; s2<njkm; s2++)
		{
			if (abs(normJKM(s1,s2)) > small)
			{
				checknorm << "# <JKM| : "    << s1 << endl;
				checknorm << "# |J'K'M'> : " << s2 << endl;
				checknorm << "# Norm: "      << normJKM(s1,s2) << endl;
				checknorm <<""<< endl;
			}
		}
	}
	checknorm.close();
}

void get_QuantumNumList_NonLinear_RealBasis(int jmax, matrix &jkmList_real,matrix &jkemList_real,matrix &jkomList_real)
{
	/* See ``Appendix: Real basis of non-linear rotor'' in Rep. Prog. Phys. vol. 77 page- 046601 (2014). */

	int j, k, m;
	int jtempcounter = 0;
	for (int j=0; j<(jmax+1); j++)
	{
		k=0; m=0;

		jkmList_real(jtempcounter,0)=j;
		jkmList_real(jtempcounter,1)=k;
		jkmList_real(jtempcounter,2)=m;
		jtempcounter=jtempcounter+1;

		for (m=1; m<(j+1); m++)
		{

			jkmList_real(jtempcounter,0)=j;
			jkmList_real(jtempcounter,1)=k;
			jkmList_real(jtempcounter,2)=m;
			jtempcounter=jtempcounter+1;
			jkmList_real(jtempcounter,0)=j;
			jkmList_real(jtempcounter,1)=k;
			jkmList_real(jtempcounter,2)=m;
			jtempcounter=jtempcounter+1;
		}

		for (k=1; k<(j+1); k++)
		{
			for (m=-j; m<(j+1); m++)
			{
				jkmList_real(jtempcounter,0)=j;
				jkmList_real(jtempcounter,1)=k;
				jkmList_real(jtempcounter,2)=m;
				jtempcounter=jtempcounter+1;
				jkmList_real(jtempcounter,0)=j;
				jkmList_real(jtempcounter,1)=k;
				jkmList_real(jtempcounter,2)=m;
				jtempcounter=jtempcounter+1;
			}
		}
	}
}

cmatrix get_umat(int njkm, matrix &wf_complex, matrix &wf_real)
{
	int size_basis = njkm*njkm;
	cmatrix u(size_basis,size_basis);
  
	// oprate with first rotor
#pragma omp parallel for 
	for (int i2=0;i2<njkm;i2++) {
		for (int i1=0;i1<njkm;i1++) {
			for (int i1p=0;i1p<njkm;i1p++) {
				u(i1*njkm+i2)+=wf_complex(i1,i1p)*wf_real(i1p,i2);
			}
		}
	}

	// oprate with second rotor
#pragma omp parallel for 
	for (int i1=0;i1<njkm;i1++) {
		for (int i2=0;i2<njkm;i2++) {
			for (int i2p=0;i2p<njkm;i2p++) {
				u(i1*njkm+i2)+=wf_complex(i2,i2p)*wf_real(i2p,i1);
			}
		}
	}

	return u;
}
