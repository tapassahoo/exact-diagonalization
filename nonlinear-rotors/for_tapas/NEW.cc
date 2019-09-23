#include <complex>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <assert.h>
#include <math.h>
#include <cmath>
#include "gauss_legendre.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <omp.h>
#ifndef PI
#define PI 3.1415926535897932384626433832795028841971693993751
#endif
#ifndef FABS
#define FABS(a) ((a)>=0?(a):-(a))
#endif

using namespace std;
int omp_get_num_threads(void);
extern "C" void gauleg(double x1,double x2,double *x,double *w,int n);
extern "C" void dsyev_( char* jobz, char* uplo, int* n, double* a, int* lda, double* w, double* work, int* lwork, int* info );
extern double **doubleMatrix(long row, long col);
extern void print_matrix( char* desc, int m, int n, double* a, int lda );
extern long double W3j(int j1, int j2,int j3,int m1,int m2,int m3);
//extern double enHFC60(double R,double *AngleL, double *AngleJ);
//extern double matrot(double *Angle,double rotmat);
extern double delta(int n, int m);
extern double doublefact(int n);
//extern double Laguerre(int n,double a,double x);
//extern double plgndr(int l,int m,double x);
//extern double plm(int l,int m,double x);
//extern double normPlm(int l,int m,double x);
extern void MatrixDiag(double* a,int n, double *w, double *b, int flag);
extern void MatrixDiagPOS(double* a,int n, double *w, double *b, int flag);
//extern void dipHFC60(double R,double *AngleL, double *AngleJ, double *dip);
//extern void InitBasis(int Kmax, int Lmax, int sizeL, int Ngrid, double *xyzgrid,double *xyzvec,double nu, double ***Basis);
//extern void InitBasisNew(int Kmax, int Lmax, int sizeL, double nu, double **BasL);
//extern void InitBasis(int Kmax, int Lmax, int sizeL, double Rmax, int Ngrid, double nu, double ***Basis);
extern void InitNumb(int Kmax,int Lmax,int Jmax,int *K1,int *L1,int *ML1,int *N1,int *J1,int *MJ1);
extern void InitSize(int Kmax,int Lmax,int Jmax,int &sizeL,int &sizeJ);
extern void RotEnergy(int Jmax, int sizeJ, double B, double D, double *Hrot);
extern void HbarEnergy(int *N1,int sizeL,int sizeJ,double hbar,double omega,double jpcm,double *Hbar);
//extern void HOEnergy(double dr,int Ngrid,int sizeL,int sizeJ,double omega,double massHF,double jpcm,int *K1,int *L1,int *ML1,double ***BasLL,double **HO);
//extern void HOEnergy(int Ngrid,double *xyzgrid,double *xyzvec,int sizeL,int sizeJ,long double omega,long double massHF,long double jpcm,int *K1,int *L1,int *ML1,double ***BasLL,double **HO);
extern void HOEnergyNew(int sizeL,int sizeJ,double omega,double massHF,double nu,double jpcm,int *K1,int *L1,int *ML1,double **HO);
extern void NumbExpanCoef(int npot,int *pV1,int *pV2, int *pV, int *mp, int &nV);
//extern void CalcCoef(double dr,int Ngrid,int npot,int nV,int *pV1,int *pV2,int *pV,int *mp,int *VL,int *VML,int *VJ,int *VMJ,int *VS,int *VMS,double **coef);
//extern void CalcCoef(int Ngrid,double *xyzgrid,int npot,int nV,int *pV1,int *pV2,int *pV,int *mp,int *VL,int *VML,int *VJ,int *VMJ,int *VS,int *VMS,double **coef);
extern void CalcCoefNew(int npot,int nV,int sizeL, int sizeJ,double nu,int *K1,int *L1,int *ML1,int *pV1,int *pV2,int *pV,int *mp,int *VL,int *VML,int *VJ,int *VMJ,int *VS,int *VMS,double ***V);
long double Binomial(double a,double b);
int main()
{
	//////////INPUT PARAMETERS/////////////////////////

	int test=0;   //if test=0 -> full calculations, if test=1 -> V =isome constatnt
	int Jmax=6;
	int Kmax=3;
	int Lmax=6;

	cout<<" Jmax="<<Jmax<<"  Kmax="<<Kmax<<"  Lmax="<<Lmax<<endl;

	double Tmax=1.;
	double Tstep=1.;
	////rHF=0.9255214817!r0(HF) J.opt.soc.Am. 57(12), 1464 (1967)//////     
	double B0HF= 20.561/1.32;	//        double D0HF=0.00213;    /
	double D0HF=0.05;//0.00213; //cm-1                                /
	double H0HF=1.418*pow(10.0,(-7.0));//cm-1  Not using at the moment/
        ///////////////////////////////////////////////////////////////////
	double WTOK=0.695035;//cm-1 ->K
	double hbar = 6.62606957*pow(10.,-34.)/(2.0*PI);
	double jpcm = 1.98630*pow(10.,-23.);//J/cm-1
	double massH=1.00794;
	double massF=18.998403;
	double massHFamu=massH+massF;
	double massHF = massHFamu*1.66054*pow(10.,-27.);//kg
        double omega=1.211*pow(10.,13.);//2nd derivative of A0000(R) at R=0 ;
        double nu=omega*massHF/(2.0*hbar);//nu needed for the translational basis

         cout<<" beta = 2*nu/10^18 "<<2.*nu*pow(10.,-18.)<<  "  nu "<<nu<<endl;
        ////Effective bond length////////////////////////////////////////////
	double B0au=B0HF/219474.63; //B0 in hartree
	double mred=massH*massF/((massH+massF)*0.0005485865);//reduced mass of HF in au
	double r0=sqrt(1./(2.0*mred*B0au))*0.529177;//new r0 in Angstrom
        ////////////////////////////////////////////////////////////////////
	cout <<"New r0, Ang :" <<r0<< "   B0HF="<<B0HF<< "  Omega *10^(-13) Hz : "<< omega/pow(10.,13.)<<endl;
	/////////////////////////////////////////////////////
        // R   Grid points   //                            //
        int Ngrid=50;                                     //
        double Rmax=0.15;//in nm (maximum distance from the center of the cage is 1.5 Ang)
        double dr = Rmax/Ngrid;//step in grid              //
        cout<<" Integration drid:  Ngrid   "<< Ngrid<<endl;//
        /////////////////////////////////////////////////////
        double xyzgrid[Ngrid];
        double xyzvec[Ngrid];
        ifstream xfile;
        ifstream wfile;  
        xfile.open("reslag_x.txt");
        wfile.open("reslag_w.txt");
        for (int i=0;i<Ngrid;i++)
        {
        double xyz;
        double xyzv;
        xfile>>xyz;
        wfile>>xyzv;
        xyzvec[i]=(xyzv);
        xyzgrid[i]=sqrt(xyz/(2.*nu))*pow(10.,9.);//*pow(10.,9.);
//        cout<<xyzgrid[i]<<" "<<xyzvec[i]<<endl;
        }    
        xfile.close();
        wfile.close();

	clock_t startcputime,endcputime;
	clock_t startV,endV;
	clock_t startMatr,endMatr;
	clock_t startEigen,endEigen;

	startcputime = clock();
        ///////////////////////////////////////////////////////////
        ///Checking the OpenMP compilation/////////////////////////
# ifdef _OPENMP                                                  //
	printf("Compiled by an OpenMP-compliant implementation.\n");
# endif                                                          //  
	int NThreads;                                            //  
#pragma omp parallel num_threads(10)                             //
	{                                                        //
		NThreads=omp_get_num_threads();                  //
	} // end omp                                             //
                                                                 //
	cout<<"NThreads="<<NThreads<<endl;                       //
                                                                 //
	///////////////////////////////////////////////////////////

	int sizeL;
	int sizeJ;
	InitSize(Kmax,Lmax,Jmax,sizeL,sizeJ); //finding the size of translational & rotational bases
	int sizetot=sizeJ*sizeL;//total size of basis functions (dot product)
	int *K1 =new int [sizetot];
	int *L1 =new int [sizetot];
	int *N1 =new int [sizetot];
	int *ML1=new int [sizetot];
	int *J1 =new int [sizetot];
	int *MJ1=new int [sizetot];
	double *Hrot=new double [sizeJ];//rotational levels of HF
	double *Hbar=new double [sizeL];//kinetic energy of HO
	double **HO;                    //HO energy to subtract 
	HO=doubleMatrix(sizeL,sizeL);

//	double ***BasLL; //Translational Basis
//	BasLL= new double **[sizeL];
//	for (int i =0; i<sizeL;i++)
//		BasLL[i]=doubleMatrix (sizeL,Ngrid);
//        double **BasL;
//        BasL=doubleMatrix(sizeL,Ngrid);   
	InitNumb(Kmax,Lmax,Jmax,K1,L1,ML1,N1,J1,MJ1);//numbering the basis functions
//        InitBasis(Kmax,Lmax,sizeL,Ngrid,xyzgrid,xyzvec,nu,BasLL);//Initialisation of the basis set for radial grid points
//        InitBasisNew(Kmax,Lmax,sizeL,nu,BasL);
//	InitBasis(Kmax,Lmax,sizeL,Rmax,Ngrid,nu,BasLL);//Initialisation of the basis set for radial grid points
	RotEnergy(Jmax,sizeJ,B0HF,D0HF,Hrot);//Rotational energy of HF
	HbarEnergy(N1,sizeL,sizeJ,hbar,omega,jpcm,Hbar);//HARMONIC OSCILLATOR ENERGIES (kinetir energy)/////
//	HOEnergy(dr,Ngrid,sizeL,sizeJ,omega,massHF,jpcm,K1,L1,ML1,BasLL,HO);//HARMONIC OSCILLATOR ENERGIES (for deltaV energy) for subtraction/////
//        HOEnergy(Ngrid,xyzgrid,xyzvec,sizeL,sizeJ,omega,massHF,jpcm,K1,L1,ML1,BasLL,HO);//HARMONIC OSCILLATOR ENERGIES (for deltaV energy) for subtraction/////
        HOEnergyNew(sizeL,sizeJ,omega,massHF,nu,jpcm,K1,L1,ML1,HO);//HARMONIC OSCILLATOR ENERGIES (for deltaV energy) for subtraction/////

	int npot=8;//number of expansion coefficicents 
	int pV1[npot],pV2[npot],pV[npot],mp[npot];
	int nV;
	NumbExpanCoef(npot,pV1,pV2,pV,mp,nV);
//	double **coef;
//	coef = doubleMatrix(nV,Ngrid);
        double ***V;
        V= new double **[nV];
        for (int i =0; i<nV;i++)
                V[i]=doubleMatrix (sizeL,sizeL);
//The numbers L ML J MJ Lambda MLambda  for expansion of the potential A^LJ_LambdaMlambda
	int *VL=new int [nV]; 
	int *VML=new int [nV];
	int *VJ=new int [nV];
	int *VMJ=new int [nV];
	int *VS=new int [nV];
	int *VMS=new int [nV];
//Precalculation the expansion coefficicnets for radial grid ponts needed for integration
//	CalcCoef(dr,Ngrid,npot,nV,pV1,pV2,pV,mp,VL,VML,VJ,VMJ,VS,VMS,coef);
//Precalculation the expansion coefficicnets for radial grid ponts needed for integration
//      CalcCoef(Ngrid,xyzgrid,npot,nV,pV1,pV2,pV,mp,VL,VML,VJ,VMJ,VS,VMS,coef);
        CalcCoefNew(npot,nV,sizeL,sizeJ,nu,K1,L1,ML1,pV1,pV2,pV,mp,VL,VML,VJ,VMJ,VS,VMS,V);
/////// ANALYTICAL <klmjmj|V|k'l'm'j'm'j> ///////////////////////////////////////////
	double **VV;
	VV=doubleMatrix(sizetot,sizetot);
	cout<<" ANALYTICAL <klmjmj|V|k'l'm'j'm'j>" <<endl; 
	for (int npos1 = 0 ; npos1 <sizeL;npos1++) //translational
		for (int npos2 = 0 ; npos2 <sizeL;npos2++)
			for (int poss1 = 0; poss1 <sizeJ;poss1++) //rotational
				for (int poss2 = 0; poss2 <sizeJ;poss2++)
				{
					int POS1=npos1*sizeJ+poss1;
					int POS2=npos2*sizeJ+poss2;
					VV[POS1][POS2]=0.;
                                     for (int ic1=0;ic1<nV;ic1++)
                                     {
						long double fact=powl((long double)(-1.),(long double)(ML1[POS2]+MJ1[POS2]))*W3j(L1[POS1],L1[POS2],VL[ic1],0,0,0)*W3j(L1[POS1],L1[POS2],VL[ic1],ML1[POS1],-ML1[POS2],VML[ic1])*W3j(J1[POS1],J1[POS2],VJ[ic1],0,0,0)*W3j(J1[POS1],J1[POS2],VJ[ic1],MJ1[POS1],-MJ1[POS2],VMJ[ic1])*sqrtl((long double)((2*L1[POS1]+1)*(2*L1[POS2]+1)*(2*VL[ic1]+1)*(2*J1[POS1]+1)*(2*J1[POS2]+1)*(2*VJ[ic1]+1)/4.));

						VV[POS1][POS2]+=V[ic1][npos1][npos2]*fact;
                                     
//						if (fact!=0.&&V[ic1][npos1][npos2]!=0.)
//						{
//							printf("POS1  %d   POS2  %d   K1 %d L1 %d  K2 %d L2 %d   V : %0.15f  fact : %0.15f   VV : %0.15f \n" ,POS1,POS2,K1[POS1],L1[POS1],K1[POS2],K1[POS2],V[ic1][npos1][npos2],fact,V[ic1][npos1][npos2]*fact);
//						}
                                      }
// printf("before POS1  %d   POS2  %d   K1 %d L1 %d  ML1 %d J1 %d MJ1 %d    K2 %d L2 %d ML2 %d J2 %d MJ2 %d    VV : %0.15f \n" ,POS1,POS2,K1[POS1],L1[POS1],ML1[POS1],J1[POS1],MJ1[POS1],K1[POS2],L1[POS2],ML1[POS2],J1[POS2],MJ1[POS2],VV[POS1][POS2]);
				//						cout<<POS1<<" "<<POS2<<" "<<VV[POS1][POS2]<<endl;
										if (npos1==npos2 && poss1==poss2 )
										{
										VV[POS1][POS2]+=Hrot[poss1]; //plus rotational energy
										VV[POS1][POS2]+=Hbar[npos1];//plus kinetic energy from HO
										}
										if (poss1==poss2)
										{
										VV[POS1][POS2] -=HO[npos1][npos2];//minus energy from HO
										}
					

// printf("after POS1  %d   POS2  %d   K1 %d L1 %d  ML1 %d J1 %d MJ1 %d    K2 %d L2 %d ML2 %d J2 %d MJ2 %d    VV : %0.15f \n" ,POS1,POS2,K1[POS1],L1[POS1],ML1[POS1],J1[POS1],MJ1[POS1],K1[POS2],L1[POS2],ML1[POS2],J1[POS2],MJ1[POS2],VV[POS1][POS2]);
				}




	delete [] VL;
	delete [] VML;
	delete [] VJ;
	delete [] VMJ;
	delete [] VS;
	delete [] VMS;


	//////////////////////

	/*
	   startV = clock();
	// Precalculating the potential and dipole moment
	double ** res_dipx;
	double ** res_dipy;
	double ** res_dipz;
	double ** res_R;
	res_dipz =doubleMatrix(sizetot,sizetot);
	res_dipy =doubleMatrix(sizetot,sizetot);
	res_dipx =doubleMatrix(sizetot,sizetot);
	res_R =doubleMatrix(sizetot,sizetot);

	double **V;
	double **Dz;
	double **Dy;
	double **Dx;
	V = doubleMatrix(Ngrid*ngl*nphi,ngl*nphi);
	Dz= doubleMatrix(Ngrid*ngl*nphi,ngl*nphi);
	Dx=doubleMatrix(Ngrid*ngl*nphi,ngl*nphi);
	Dy=doubleMatrix(Ngrid*ngl*nphi,ngl*nphi); 
	double *dip;
	dip=new double [3];

	if (test ==0)
	{ 
	for (int ix = 0; ix < Ngrid; ix++)
	for (int ith = 0; ith < ngl; ith++)
	for (int iph = 0; iph < nphi; iph++)
	for (int i=0;i<ngl;i++)
	for (int j=0;j<nphi;j++)
	{
	int pppL=ix*ngl*nphi+ith*nphi+iph;
	int pppJ=i*nphi+j;
	R=xyzgrid[ix]*10.;//from nm to Ang (r in PES is in Ang)
	AngleL[1]=xphi[iph];
	AngleL[0]=acos(xgl[ith]);
	AngleJ[1]=xphi[j];
	AngleJ[0]=acos(xgl[i]);

	dipHFC60(R,AngleL,AngleJ,dip);

	V[pppL][pppJ]=enHFC60(R,AngleL,AngleJ);
	Dx[pppL][pppJ]=dip[0];
	Dy[pppL][pppJ]=dip[1];
	Dz[pppL][pppJ]=dip[2];        
	//        cout<< R<<" "<<AngleL[0]<<" "<<AngleL[1]<<" "<< AngleJ[0]<<" "<< AngleJ[1]<<" "<<V[pppL][pppJ]<<" "<<Dz[pppL][pppJ]<<endl;
	}

	cout<< " Evaluation of the PES is finished:   size="<<Ngrid*ngl*nphi*ngl*nphi<<"   points ";
	}
	endV = clock();
	cout<< "it took:  "<<-(startV-endV)<<"  s of CPU"<<endl;
	delete [] dip;
	//////////////////////////////MATRIX ELEMENTS OF THE POTENTIAL////////////////////////////////////////

	startMatr = clock();

	cout<<" QUADRATURE integration"<<endl;
	double  *res1;                 
	double  *resx;
	double  *resy;
	double  *resz; 
	double  *res0;
	res1 = new double [ngl*nphi]; 
	resx = new double [ngl*nphi];
	resy = new double [ngl*nphi];
	resz = new double [ngl*nphi];
	res0 = new double [ngl*nphi];
	int count =0;
	for (int npos1 = 0 ; npos1 <sizeL;npos1++) //translational
	for (int npos2 = 0 ; npos2 <sizeL;npos2++)
	for (int poss1 = 0; poss1 <sizeJ;poss1++) //rotational
		for (int poss2 = 0; poss2 <sizeJ;poss2++) 
		{ 

			int POS1=npos1*sizeJ+poss1;
			int POS2=npos2*sizeJ+poss2;        

			int ni=2*KK[npos1]+LL[npos1];
			int nj=2*KK[npos1]+LL[npos1];

			res[POS1][POS2]=0.;
			res_dipx[POS1][POS2]=0.;
			res_dipy[POS1][POS2]=0.;
			res_dipz[POS1][POS2]=0.;
			res_R[POS1][POS2]= 0.;
			double resultat =  0.;
			double resdipx  =  0.;
			double resdipy  =  0.;
			double resdipz  =  0.;
			double resR     =  0.;
#pragma omp parallel for reduction(+: resultat,resdipx,resdipy,resdipz)
			for (int i=0;i<ngl;i++)          
				for (int j=0;j<nphi;j++)         
				{                                
					int pppJ=i*nphi+j;               
					res1[pppJ]=0.;  
					resx[pppJ]=0.;                 
					resy[pppJ]=0.;
					resz[pppJ]=0.;
					res0[pppJ]=0.;
					for (int ix = 0; ix < Ngrid; ix++)
						for (int ith = 0; ith < ngl; ith++)
							for (int iph = 0; iph < nphi; iph++)
							{
								int pppL=ix*ngl*nphi+ith*nphi+iph;

								if (test ==1)
									res1[pppJ]+=BasL[npos1][npos2][pppL];

								if (test == 0)
								{       
									resx[pppJ]+=Dx[pppL][pppJ]*BasL[npos1][npos2][pppL];
									resy[pppJ]+=Dy[pppL][pppJ]*BasL[npos1][npos2][pppL];
									resz[pppJ]+=Dz[pppL][pppJ]*BasL[npos1][npos2][pppL]; 
									res1[pppJ]+=V[pppL][pppJ]*BasL[npos1][npos2][pppL];
									res0[pppJ]+=xyzgrid[ix]*10.*BasL[npos1][npos2][pppL];
								} //endif test=0
							}
					if (abs(res1[pppJ])>pow(10.,-16))
					{
						resultat+=res1[pppJ]*BasRot[pppJ][poss1][poss2]; 
						resdipx  +=resx[pppJ]*BasRot[pppJ][poss1][poss2];
						resdipy  +=resy[pppJ]*BasRot[pppJ][poss1][poss2];
						resdipz  +=resz[pppJ]*BasRot[pppJ][poss1][poss2];
						resR     +=res0[pppJ]*BasRot[pppJ][poss1][poss2];
					}
				} //integration

			cout<<"POS1 " <<POS1<<"  POS2 " <<POS2<<"  "<<resultat<<endl;

			// if (POS1 == 0)   
			// cout<<"POS  :" <<POS1<<"  "<<npos1<<"  "<<poss1<<"       POS': " <<POS2<<"  "<<npos2<<"  "<<poss2<<"    "<<resultat<<"    Dip:   "<<resdipx<<" "<<resdipy<<" "<<resdipz<<" R :"<<resR<<endl;

			count +=1;

			if (count%1000 == 0)     
				cout<<"done "<<count<<"  out of "<< sizetot*sizetot  <<endl;  

			res_dipx[POS1][POS2]= resdipx;
			res_dipy[POS1][POS2]= resdipy;
			res_dipz[POS1][POS2]= resdipz;
			res[POS1][POS2] = resultat;
			res_R[POS1][POS2]= resR;
			//       cout<<" POS1 "<< POS1<<" POS2 "<<POS2<<"  res ="<<res[POS1][POS2]<<endl;
			if (npos1==npos2 )
				res[POS1][POS2]+=Hrot[poss1][poss2]; //plus rotational energy


			if (npos1==npos2 & poss1==poss2 )
				res[POS1][POS2]+=Hbar[npos1];//plus kinetic energy from HO

			if (poss1==poss2 & npos1==npos2)
			{
				res[POS1][POS2] -=nqsqnp[npos1][npos2];
			}


			//        cout<<" "<<POS1<<" " <<POS2<<" "<<res[POS1][POS2]<<endl;

		}//loop over basis
	endMatr = clock();
	cout<< "Calculation of matrix elements took:  "<<-(startMatr-endMatr)<<"  s of CPU"<<endl;

	*/


		delete [] Hbar;
	delete [] Hrot;
	//////////////////Eignestates///////////////////////
	startEigen = clock();

	double *H0;
	double *H1;
	H0=new double [sizetot*sizetot];
	H1=new double [sizetot*sizetot];
	int i1=0;
	for (int i = 0; i <sizetot; i++)
		for (int j = 0; j <sizetot; j++)
		{
			H0[i1]= VV[i][j];;
			i1 +=1;
		}

	double *En;
	double *vec;
	double **Evec;
	double **Evec1;
	double *En1;
	double *vec1;
	En=new double [sizetot];
	En1=new double [sizetot];
	vec1=new double [sizetot*sizetot]; 
	vec=new double [sizetot*sizetot];
	Evec=doubleMatrix(sizetot,sizetot);
	Evec1=doubleMatrix(sizetot,sizetot);
	MatrixDiagPOS(H0,sizetot,En,vec,0);
	endEigen = clock();
	cout<< "Diagonalization took:  "<<-(startEigen-endEigen)<<"  s of CPU"<<endl;

	/////////////Stroing Eigen Vectors///////////////////////////////////////

	int k0=-1;
	for (int i=0;i<sizetot;i++)
		for (int nn=0;nn<sizetot;nn++)
		{
			k0+=1;
			Evec[nn][i] = vec[k0];
		}
	delete [] vec;
	delete [] vec1;
	////////FINDING LEADING CONTRIBUTION TO THE I STATE WAVE FUNCTION////////
	FILE * myfile;
	myfile=fopen("coef","w"); 


	int NN1[sizetot];
	int JJ1[sizetot];
	int LL1[sizetot];


	double COEF[sizetot];
	cout<<" LEADING CONTRIBUTION TO THE I STATE WAVE FUNCTION "<<endl;
	for (int nn=0;nn<sizetot;nn++)
	{
		double Eps=0.  ;
		double maxEps =0.;
		int n0 = 0;
		fprintf(myfile," STATE : %d \n",nn);
		fprintf(myfile,"   C^2       n j l      |   k l ml j mj\n");
		for (int i=0;i<sizetot;i++)
		{
			Eps+=Evec[nn][i]*Evec[nn][i];

			if ((Evec[nn][i]*Evec[nn][i])>maxEps) 
			{
				maxEps=Evec[nn][i]*Evec[nn][i]  ;
				n0=i;
			}
			fprintf(myfile," %f    %d %d %d      |   %d %d %d %d %d\n",Evec[nn][i]*Evec[nn][i],N1[i],J1[i],L1[i],K1[i],L1[i],ML1[i],J1[i],MJ1[i]);
		}
		COEF[nn]=maxEps; 
		NN1[nn]=N1[n0];
		JJ1[nn]=J1[n0];
		LL1[nn]=L1[n0];
//		cout<<" level "<<nn<<" EPStot="<<Eps<<" Max contribution from ( n j l ): "<<N1[n0]<<" "<<J1[n0]<<" "<<L1[n0]<< " with |eps|^2 : " <<maxEps<<endl;
	}
	fclose(myfile);

	delete [] K1;
	delete [] L1;
	delete [] ML1;
	delete [] J1;
	delete [] MJ1;       

	///////////////////SOS/////////////////////////

	cout<<" <Etot>  relative to the ground state   "<<endl;
	cout<< "T (K)     <Etot> (cm-1)  "<<endl;
	double Enav=0.,Zav=0.;
	for (double T=Tmax;T<Tmax+1.;T+=Tstep)
	{
#pragma omp parallel for reduction(+: Enav,Zav)
		for (int i=0;i<sizetot;i++)
		{
			Enav +=(En[i]-En[0])*exp(-(En[i]-En[0])/(WTOK*T));
			Zav  +=exp(-(En[i]-En[0])/(WTOK*T));
		}
		cout<< T<<"    "<<Enav/Zav<<endl;
	}

	//////////////////////////////////////////////


	/*
	   double *intensityx;
	   double *intensityy;
	   double *intensityz;
	   double *Rav;
	   intensityx = new double [sizetot];
	   intensityy = new double [sizetot];
	   intensityz = new double [sizetot];
	   Rav        = new double [sizetot];
	/////CALCULATION OF <i|R|i>///
	cout<< " Calculation of average < i | R | i> "<<endl;
	for (int i =0; i<sizetot;i++)
	{
	Rav[i]=0;
	for (int pos1=0;pos1<sizetot;pos1++)
	for (int pos2=0;pos2<sizetot;pos2++)
	{
	Rav[i] +=  Evec[i][pos1]*Evec[i][pos2]*res_R[pos1][pos2];
	}
	cout<< " i :" <<i<<"  <i| R |i> = "<<Rav[i]<<endl; 
	}
	delete [] Rav;

	//        cout<< " Intensity of transitions out of the ground state  "<<endl;
	double T=5.0;
	for (int i =0; i<sizetot;i++)
	{
	intensityx[i]=0.;
	intensityy[i]=0.;
	intensityz[i]=0.;
	for (int pos1=0;pos1<sizetot;pos1++)
	for (int pos2=0;pos2<sizetot;pos2++)
	{
	intensityx[i] += (Evec[0][pos1]*Evec[i][pos2]*res_dipx[pos1][pos2]*res_dipx[pos1][pos2]*Evec[0][pos1]*Evec[i][pos2]);
	intensityy[i] += (Evec[0][pos1]*Evec[i][pos2]*res_dipy[pos1][pos2]*res_dipy[pos1][pos2]*Evec[0][pos1]*Evec[i][pos2]);
	intensityz[i] += (Evec[0][pos1]*Evec[i][pos2]*res_dipz[pos1][pos2]*res_dipz[pos1][pos2]*Evec[0][pos1]*Evec[i][pos2]);
	}
	printf(" level  %d  |<0| mu |i>|^2 :  %f \n",i, (intensityx[i]+intensityy[i]+intensityz[i])/3.0);
	if (i==0) 
	{intensityx[i]=0.;
	intensityy[i]=0.;
	intensityz[i]=0.;}
	else
	{ 
	intensityx[i] *=(En[i]-En[0])*(exp(-(En[0]-En[0])/(WTOK*T))-exp(-(En[i]-En[0])/(WTOK*T)))/Zav;
	intensityy[i] *=(En[i]-En[0])*(exp(-(En[0]-En[0])/(WTOK*T))-exp(-(En[i]-En[0])/(WTOK*T)))/Zav;
	intensityz[i] *=(En[i]-En[0])*(exp(-(En[0]-En[0])/(WTOK*T))-exp(-(En[i]-En[0])/(WTOK*T)))/Zav;}
	}
	 */
	delete [] Evec;
	delete [] Evec1;
	delete [] H0;
	delete [] H1;

	for (int i = 0; i<sizetot;i++)//sizetot;i++)
	{
		//                printf(" level  %d  dEan %f  dEqua %f intensity : %f \n",i,En[i]-En[0],En1[i]-En1[0], (intensityx[i]+intensityy[i]+intensityz[i])/3.0);
		printf(" level  %d  E  %0.16f  dEan %0.16f    ( n j l )=( %d %d %d )  Eps^2 =%0.5f\n",i,En[i],En[i]-En[0],NN1[i],JJ1[i],LL1[i],COEF[i]); 
	}

	//	delete [] intensityx;
	//	delete [] intensityy;
	//	delete [] intensityz;
	delete [] En;
	delete [] En1; 

	endcputime = clock();
	cout << " It took " << endcputime - startcputime << " s of CPU "<<endl;



}//end program main



//////////////////////////////////////////////


//Interaction energy of one rotor - fixed dipole at R=10.05Ang
double enDD(double theta)
{
	double dipH2O=0.51/2.54177; //reduced dipole of H2O mu=0.51+-0.5 Debye: Nature Communications 6, 8112 (2015), doi 10.1038/ncomms9112.
	double dd2 = dipH2O *dipH2O;
	double a0 = 0.529177, conv = a0 * a0 * a0 * 315775.13;
	double R = 10.05, R3 =R*R*R ;//fixed molecule is on the right side from the molecule under consideration

	return dd2*conv*(-2.0*cos(theta)/R3);
}
double **doubleMatrix(long row, long col)
	// based on nrutils.c dmatrix
{
	double **mat_;
	if (row > 0) {
		mat_ = new double*[row];
	} else {
		// free_doubleMatrix requires at least one element
		mat_ = new double*[1];
	}
	mat_[0] = new double [row*col];
	assert(mat_    != NULL);
	assert(mat_[0] != NULL);
	for (long i=1; i<row; i++)
		mat_[i] = mat_[i-1]+col;
	return mat_;
}

//delta function
double delta(int n, int m)
{
	if(n==m)
		return 1.0;
	else
		return 0.0;
}


//Associated not normalized Legendre polynomials for positive m only (from numerical recipies)
double plgndr(int l,int m,double x){
	if( (m < 0) || (m > l) || (x > 1.) || (x < -1.))
		return 0;
	else
	{
		double pmm=1.; 
		if (m > 0)
		{
			double somx2=sqrt((1.-x)*(1.+x));
			double fact=1.;
			for (int i=1;i<=m;i++) {
				pmm *=-fact*somx2;
				fact +=2.;  }//end loop i
		}//end if m>0
		if(l == m)
			return pmm;
		else
		{
			double pmmp1=x*(2*m+1)*pmm; //!Compute P m m+1.
			if(l == (m+1))
				return pmmp1;
			else //!Compute P ml , l>m + 1.
			{
				double pll;
				for (int ll=m+2; ll<=l;ll++)  {
					pll=(x*(2*ll-1)*pmmp1-(ll+m-1)*pmm)/(ll-m);
					pmm=pmmp1;
					pmmp1=pll;  }//end loop ll
				return pll;
			}
		}
	} //endif m>=0
}//end functopn plnldr


//Associated not normalized Legendre polynomials for any m
double plm(int l,int m,double x){
	if (m > 0 || (m == 0))
		return plgndr(l,m,x);
	else {
		double fact= exp(lgamma((double)(l-abs(m)+1))-lgamma((double)(l+abs(m)+1)));
		double rr=pow(-1,abs(m))*plgndr(l,abs(m),x)*fact;
		return rr;   }} //end function plm

		//Normalized Associated Legendre Polynomials
double normPlm(int l,int m,double x)
{
	double norm=pow(-1.,(double)m)*sqrt((double)l+0.5)*sqrt(tgamma(double(l-m+1))/tgamma(double(l+m+1)));
	return norm*plm(l,m,x);
}

long double W3j(int j1, int j2,int j3,int m1,int m2,int m3)
{
	if ((abs(j1-j2)>j3) || (j3 > (j1+j2)) || ((m1+m2+m3) !=0) || (abs(m1) > j1 )||(abs(m2)>j2)||(abs(m3)>j3))
	{
		return 0.;
	}
	int    t1 = j2 - m1 - j3;
	int    t2 = j1 + m2 - j3;
	int    t3 = j1 + j2 - j3;
	int    t4 = j1 - m1;
	int    t5 = j2 + m2;

	int    tmin = max( 0, max( t1, t2 ) );
	int    tmax = min( t3, min( t4, t5 ) );

	long double    wigner = 0.;

	for (int t=tmin;t<tmax+1;t++)
	{
		long double fact1=( lgammal((long double)(t+1)) + lgammal((long double)(t-t1+1)) + lgammal((long double)(t-t2+1)) + lgammal((long double)(t3-t+1)) + lgammal((long double)(t4-t+1)) + lgammal((long double)(t5-t+1)) );
		wigner += powl(-(long double)1,(long double)t) / expl(fact1);
	}

	long double fact2 = lgammal((long double)(j1+j2-j3+1)) + lgammal((long double)(j1-j2+j3+1)) + lgammal((long double)(-j1+j2+j3+1))+ lgammal((long double)(j1+m1+1)) + lgammal((long double)(j1-m1+1)) + lgammal((long double)(j2+m2+1)) + lgammal((long double)(j2-m2+1)) + lgammal((long double)(j3+m3+1)) + lgammal((long double)(j3-m3+1)) - lgammal((long double)(j1+j2+j3+1+1)); 

	return wigner * powl(-(long double)1,(long double)(j1-j2-m3)) * sqrt( expl(fact2));



}



/*double W3j(int j1, int j2,int j3,int m1,int m2,int m3)
  {
  if ((abs(j1-j2)>j3) || (j3 > (j1+j2)) || ((m1+m2+m3) !=0) || (abs(m1) > j1 )||(abs(m2)>j2)||(abs(m3)>j3))
  {
//cout<<"ZERO"<<endl;
return 0.;
}
int    t1 = j2 - m1 - j3;
int    t2 = j1 + m2 - j3;
int    t3 = j1 + j2 - j3;
int    t4 = j1 - m1;
int    t5 = j2 + m2;

int    tmin = max( 0, max( t1, t2 ) );
int    tmax = min( t3, min( t4, t5 ) );

double    wigner = 0.;

for (int t=tmin;t<tmax+1;t++)
{
wigner += pow(-1,t) / ( tgamma(double(t)+1.) * tgamma(double(t-t1)+1.) * tgamma(double(t-t2)+1.) * tgamma(double(t3-t)+1.) * tgamma(double(t4-t)+1.) * tgamma(double(t5-t)+1.) );
}
return wigner * pow(-1,(j1-j2-m3)) * sqrt( tgamma(double(j1+j2-j3+1)) * tgamma(double(j1-j2+j3+1)) * tgamma(double(-j1+j2+j3+1)) / tgamma(double(j1+j2+j3+1+1)) * tgamma(double(j1+m1+1)) * tgamma(double(j1-m1+1)) * tgamma(double(j2+m2+1)) * tgamma(double(j2-m2+1)) * tgamma(double(j3+m3+1)) * tgamma(double(j3-m3+1)) );



}
 */


//Wigner D-functions (It takes Eq. 3.57 of Zare, 1988)
double wigd(int j, int m, int k, double th)
{
	double pre1,denorm,pre2,cosfac,sinfac,wigd_temp,thehlf,f1,f2,f3,f4;
	int nulow,nuup,nu;

	f1=tgamma(j+k+1);
	f2=tgamma(j-k+1);
	f3=tgamma(j+m+1);
	f4=tgamma(j-m+1);
	pre1=sqrt(f1*f2*f3*f4)  ;     
	nulow=max(0,(k-m));
	nuup=min((j+k),(j-m));
	thehlf=0.5*th;
	wigd_temp=0.;

	for (nu=nulow;nu<nuup+1;nu++)
	{
		denorm=tgamma(j-m-nu+1)*tgamma(j+k-nu+1)*tgamma(nu+m-k+1)*tgamma(nu+1)*pow(-1,nu);
		pre2=pre1/denorm;
		sinfac=pow(-sin(thehlf),(m-k+2*nu));
		cosfac=pow(cos(thehlf),(2*j+k-m-2*nu));
		wigd_temp +=pre2*cosfac*sinfac;
	}
	return wigd_temp;
}


// Auxiliary routine: printing a matrix ///
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
	int i, j;
	printf( "\n %s\n", desc );
	for( i = 0; i < m; i++ ) {
		for( j = 0; j < n; j++ ) printf( " %20.10f", a[i+j*lda] );
		printf( "\n" );
	}
}

////////////////DIAGONALISATION OF MATRIX <Hrot+V>/////////////////////////////
void MatrixDiag(double* a, int N, double *w, double *b, int flag)
{
	int LDA=N;
	int info,lwork;
	double wkopt;
	double* work;
	// w[N];
	lwork = -1;
	dsyev_( "Vectors", "Upper", &N, a, &LDA, w, &wkopt, &lwork, &info );
	lwork = (int)wkopt;
	work = (double*)malloc( lwork*sizeof(double) );
	// Solve eigenproblem //
	dsyev_( "Vectors", "Upper", &N, a, &LDA, w, work, &lwork, &info );
	// Check for convergence //
	if( info > 0 ) {
		printf( "The algorithm failed to compute eigenvalues.\n" );
		exit( 1 );
	}
	// Print eigenvalues //
	if (flag == 1)
		print_matrix( "Eigenvalues of <Hrot+V>  ", 1, N, w, 1 );
	// Print eigenvectors //
	//print_matrix( "Eigenvectors (stored columnwise)", N, N, a, LDA );

	int j=0; //ground state EigenVector
	for( int i = 0; i < N; i++ ) {
		b[i]= a[i+j*N];}

	// Free workspace //
	free( (void*)work );
}

////////////////////////
void MatrixDiagPOS(double* a, int N, double *w, double *b, int flag)
{
	int LDA=N;
	int info,lwork;
	double wkopt;
	double* work;
	// w[N];
	lwork = -1;
	dsyev_( "Vectors", "Upper", &N, a, &LDA, w, &wkopt, &lwork, &info );
	lwork = (int)wkopt;
	work = (double*)malloc( lwork*sizeof(double) );
	// Solve eigenproblem //
	dsyev_( "Vectors", "Upper", &N, a, &LDA, w, work, &lwork, &info );
	// Check for convergence //
	if( info > 0 ) {
		printf( "The algorithm failed to compute eigenvalues.\n" );
		exit( 1 );
	}
	// Print eigenvalues //
	if (flag == 1)
		print_matrix( "Eigenvalues of <Hrot+V>  ", 1, N, w, 1 );
	// Print eigenvectors //
	//        print_matrix( "Eigenvectors (stored columnwise)", N, N, a, LDA );

	int k=0; 
	for( int i = 0; i < N; i++ ) 
		for (int j = 0; j < N; j++){
			b[k]= a[i+j*N];k+=1;}

	// Free workspace //
	free( (void*)work );
}
/////////////////////////////////////////


double Laguerre(int n,double a,double x)
{
	if(n==0) 
		return 1.;

	if(n==1)
	{
		return -x+a+1.;
	}

	if (n==2)
	{
		return x*x*0.5-(a+2.0)*x+(a+2.0)*(a+1.0)*0.5;
	}

	if (n==3)
	{
		return -x*x*x/6.0+(a+3.0)*x*x*0.5-(a+2.0)*(a+3.0)*x*0.5+(a+1.0)*(a+2.0)*(a+3.0)/6.0;
	}
	if(n>3)
	{
		cout <<" NOT YET PROGRAMMED "<<endl;
		return 0.;
	}
}



double doublefact(int n)
{
	int k;
	if (n%2==0)
	{
		k=n/2;
		if (k<0)
		{return 0.;}
		else
		{return pow(2.,k)*tgamma((double)(k+1));}
	}
	else
	{
		k=(n+1)/2;
		if (k<1)
		{return 0.;}
		else
		{return tgamma((double)(2.0*k+1.))/(pow(2.,k)*tgamma((double)(k+1)));}

	}
}

double enHFC60(double R, double *AngleL,double *AngleJ)
{
	double kcal2cm=4.184*83.59539;
	int n=8;
	int p1[n],p2[n],p[n],mp[n];
	double x[n];

	double R2=R*R;
	double R3=R2*R;
	double R4=R2*R2;

	p1[0]=0;
	p1[1]=0;
	p1[2]=0;
	p1[3]=1;
	p1[4]=1;
	p1[5]=1;
	p1[6]=2;
	p1[7]=2;
	p2[0]=0;
	p2[1]=6;
	p2[2]=6;
	p2[3]=1;
	p2[4]=5;
	p2[5]=5;
	p2[6]=2;
	p2[7]=4;
	p[0]=0;
	p[1]=6;
	p[2]=6;
	p[3]=0;
	p[4]=6;
	p[5]=6;
	p[6]=0;
	p[7]=6;
	mp[0]=0;
	mp[1]=0;
	mp[2]=5;
	mp[3]=0;
	mp[4]=0;
	mp[5]=5;
	mp[6]=0;
	mp[7]=0;

	x[0]= 11.6480952*R4+7.011576190*R2-13.7432 ;
	x[1]= 0.07610476*R4+0.002865809*R2+0.024668;
	x[2]=0.185607142*R4+0.104138214*R2-0.0448  ;
	x[3]= 1.93809523*R3-0.972523809*R          ;
	x[4]=-0.48571428*R3-0.108571428*R          ;
	x[5]=0.148095238*R3+0.170576190*R          ;
	x[6]=0.915595238*R4+0.100101190*R2+0.0026  ;
	x[7]=0.230009523*R3+0.048229619*R-0.000516 ;



	double EHFC60 = 0.0;
	for (int i = 0;i<n;i++)
	{
		int L = p1[i];
		int J = p2[i];
		int LL =p[i];

		double t=0.;
		int r=mp[i];
		for (int r1=-L;r1<=L;r1++)
			for (int r2=-J;r2<=J;r2++)
			{

				if ((abs(L-J)>LL) || (LL > (J+L)) || ((r1+r2+r) !=0)  )
				{t+=0.;}
				else
				{
					//cout<<L<<" "<<r1<<" "<<J<<" "<<r2<<" "<<LL<<" "<<r<<endl;
					t+=W3j(L,J,LL,r1,r2,r)*normPlm(L,r1,cos(AngleL[0]))*normPlm(J,r2,cos(AngleJ[0]))*cos(r1*AngleL[1]+r2*AngleJ[1])*pow(-1,r1+r2)/sqrt((2.0*L+1.0)*(2.0*J+1.0));
				}
			}
		EHFC60+=x[i]*t;
	}//end loop i


	return EHFC60*kcal2cm;
}

void dipHFC60(double R, double *AngleL,double *AngleJ, double *dip)
{
	int n=5;
	int p1[n],p2[n],p[n],mpx[n],mpz[n],r;
	double x[n];
	double y[n];
	double z[n];
	double R2=R*R;
	double R3=R2*R;
	double R4=R2*R2;

	p1[0]=0; 
	p1[1]=1; 
	p1[2]=1; 
	p1[3]=2; 
	p1[4]=2; 

	p2[0]=1; 
	p2[1]=0; 
	p2[2]=2; 
	p2[3]=1; 
	p2[4]=3; 

	p[0]=1;  
	p[1]=1;  
	p[2]=1;  
	p[3]=1;  
	p[4]=1;  

	mpz[0]=0;
	mpz[1]=0;
	mpz[2]=0;
	mpz[3]=0;
	mpz[4]=0;

	mpx[0]=1;
	mpx[1]=1;
	mpx[2]=1;
	mpx[3]=1;
	mpx[4]=1;

	y[0]= 0.0353704*R2+0.2757896             ;
	y[1]= 0.0456139*R3+0.0088286*R           ;
	y[2]=-0.0028807*R3+0.0300772*R           ;
	y[3]=-0.0166319*R2-0.0013999             ;
	y[4]=-0.0044504*R4-0.0015760*R2-0.0008864;

	x[0]=-y[0];
	x[1]=-y[1];
	x[2]=-y[2];
	x[3]=-y[3];
	x[4]=-y[4];

	z[0]= 0.0138312*R4-0.0255437*R2-0.1933344;
	z[1]=-0.0420837*R3-0.0061120*R           ;
	z[2]= 0.0197931*R3-0.0162010*R           ;
	z[3]=-0.0104635*R4+0.0096006*R2+0.0004281;
	z[4]= 0.0071702*R2+0.0003463             ;



	dip[0]=0.0;
	dip[1]=0.0;
	dip[2]=0.0;

	for (int i = 0;i<n;i++)
	{
		int L = p1[i];
		int J = p2[i];
		int LL =p[i];

		double tx=0.;
		double ty=0.;
		double tz=0.;
		r=mpx[i];
		for (int r1=-L;r1<=L;r1++)
			for (int r2=-J;r2<=J;r2++)
			{
				if ((abs(L-J)>LL) || (LL > (J+L)) || ((r1+r2+r) !=0)  )
				{tx+=0.;
					ty+=0.;
				}
				else
				{
					tx+=W3j(L,J,LL,r1,r2,r)*normPlm(L,r1,cos(AngleL[0]))*normPlm(J,r2,cos(AngleJ[0]))*cos(r1*AngleL[1]+r2*AngleJ[1])*pow(-1,r1+r2)/sqrt((2.0*L+1.0)*(2.0*J+1.0));
					ty+=W3j(L,J,LL,r1,r2,r)*normPlm(L,r1,cos(AngleL[0]))*normPlm(J,r2,cos(AngleJ[0]))*sin(r1*AngleL[1]+r2*AngleJ[1])*pow(-1,r1+r2)/sqrt((2.0*L+1.0)*(2.0*J+1.0));
				}
			}

		r=mpz[i];
		for (int r1=-L;r1<=L;r1++)
			for (int r2=-J;r2<=J;r2++)
			{
				if ((abs(L-J)>LL) || (LL > (J+L)) || ((r1+r2+r) !=0)  )
				{ tz+=0.;}
				else
				{
					tz+=W3j(L,J,LL,r1,r2,r)*normPlm(L,r1,cos(AngleL[0]))*normPlm(J,r2,cos(AngleJ[0]))*cos(r1*AngleL[1]+r2*AngleJ[1])*pow(-1,r1+r2)/sqrt((2.0*L+1.0)*(2.0*J+1.0));
				}
			}

		dip[0]+=x[i]*tx;
		dip[1]+=y[i]*ty;
		dip[2]+=z[i]*tz;

	}//end loop i

	dip[0]*=2.54177;
	dip[1]*=2.54177;
	dip[2]*=2.54177;

}

//void InitBasis(int Kmax, int Lmax, int sizeL, double Rmax, int Ngrid, double nu, double ***BasLL)
void InitBasis(int Kmax, int Lmax, int sizeL, int Ngrid,double *xyzgrid,double *xyzvec, double nu, double ***BasLL)
{
//	double xyzgrid[Ngrid];
//	double dr = Rmax/Ngrid; //in nm
	int K[sizeL];
	int L[sizeL];
	int ML[sizeL];

	int ps=-1;
	for (int K1=0;K1<Kmax+1;K1++)
		for (int L1=0;L1<Lmax+1;L1++)
			for (int ML1=-L1;ML1<=L1;ML1++)
			{
				ps+=1;
				K[ps]=K1;
				L[ps]=L1;
				ML[ps]=ML1;
			}

//	for (int i = 0; i <Ngrid; i++)
//	{
//		xyzgrid[i] =0.+i*dr ;//in nm
		//                cout <<xyzgrid[i]*10.<<endl;//in Ang
//	}

	for (int pos=0;pos<sizeL;pos++)
		for (int poss=0;poss<sizeL;poss++)
			for (int ix   = 0; ix   < Ngrid; ix++)
			{

				double Norm1=sqrt(sqrt(2.*nu*nu*nu/PI)*(pow(2.,(double)K[pos]+2.*(double)L[pos]+3.)*tgamma((double)(K[pos]+1))*pow(nu,(double)L[pos]))/doublefact(2*K[pos]+2*L[pos]+1));

				double Norm2=sqrt(sqrt(2.*nu*nu*nu/PI)*(pow(2.,(double)K[poss]+2.*(double)L[poss]+3.)*tgamma((double)(K[poss]+1))*pow(nu,(double)L[poss]))/doublefact(2*K[poss]+2*L[poss]+1));

//				BasLL[pos][poss][ix]=Norm1*Norm2*pow(xyzgrid[ix]/pow(10.,9.),(double)L[pos])*pow(xyzgrid[ix]/pow(10.,9.),(double)L[poss])*exp(-2.*nu*xyzgrid[ix]*xyzgrid[ix]/pow(10.,18.))*Laguerre(K[pos],float(L[pos])+0.5,(2.*nu*xyzgrid[ix]*xyzgrid[ix]/pow(10.,18.)))*Laguerre(K[poss],float(L[poss])+0.5,(2.*nu*xyzgrid[ix]*xyzgrid[ix]/pow(10.,18.)))*(xyzgrid[ix]/pow(10.,9.))*(xyzgrid[ix]/pow(10.,9.))*dr/pow(10.,9.);
 BasLL[pos][poss][ix]=Norm1*Norm2*pow(xyzgrid[ix]/pow(10.,9.),(double)L[pos])*pow(xyzgrid[ix]/pow(10.,9.),(double)L[poss])*Laguerre(K[pos],float(L[pos])+0.5,(2.*nu*xyzgrid[ix]*xyzgrid[ix]/pow(10.,18.)))*Laguerre(K[poss],float(L[poss])+0.5,(2.*nu*xyzgrid[ix]*xyzgrid[ix]/pow(10.,18.)))*xyzvec[ix]*xyzvec[ix]*exp(-2.*nu*xyzgrid[ix]*xyzgrid[ix]/pow(10.,18.))*(xyzgrid[ix]/pow(10.,9.))*(xyzgrid[ix]/pow(10.,9.));//*exp(-2.*nu*xyzgrid[ix]*xyzgrid[ix]/pow(10.,18.));//*(xyzgrid[ix]/pow(10.,9.))*(xyzgrid[ix]/pow(10.,9.));//*xyzvec[ix]*xyzvec[ix];


			}
}
void InitSize(int Kmax,int Lmax,int Jmax,int &sizeL,int &sizeJ)
{
	sizeL =0;
	for (int K=0;K<Kmax+1;K++)
		for (int L=0;L<Lmax+1;L++)
			for (int ML=-L;ML<=L;ML++)
				sizeL +=1;


	sizeJ = 0;
	for (int J=0;J<Jmax+1;J++)
		for (int M=-J;M<J+1;M++)
			sizeJ +=1;



	cout<<" InitSize done "<<sizeL<<" "<<sizeJ<<endl;

}

void InitNumb(int Kmax,int Lmax,int Jmax,int *K1,int *L1,int *ML1,int *N1,int *J1,int *MJ1)
{
	int sizeL =0;
	for (int K=0;K<Kmax+1;K++)
		for (int L=0;L<Lmax+1;L++)
			for (int ML=-L;ML<=L;ML++)
				sizeL +=1;

	int sizeJ = 0;
	for (int J=0;J<Jmax+1;J++)
		for (int M=-J;M<J+1;M++)
			sizeJ +=1;


	int KK[sizeL];
	int LL[sizeL];
	int MML[sizeL];
	int NN[sizeL];
	int JJ[sizeJ];
	int MMJ[sizeJ];
	int ps=-1;
	for (int K=0;K<Kmax+1;K++)
		for (int L=0;L<Lmax+1;L++)
			for (int ML=-L;ML<=L;ML++)
			{
				ps+=1;
				KK[ps]=K;
				LL[ps]=L;
				MML[ps]=ML;
				NN[ps]=2*K+L;
				cout<< "npos "<<ps<<" K  L  ML " <<KK[ps]<<" "<<LL[ps]<<" "<<MML[ps]<<"  N   = "<<2*K+L<<endl;

			}
	int pss=-1;
	for (int J=0;J<Jmax+1;J++)
		for (int M=-J;M<J+1;M++)
		{
			pss+=1;
			JJ[pss]=J;
			MMJ[pss]=M;
			cout<< "pos "<<pss<<" J  MJ " <<JJ[pss]<<" "<<MMJ[pss]<<endl;
		}

	//total number of levels

	cout<<"Total size="<<sizeJ*sizeL<<"    J MJ size="<<sizeJ<<"    K L ML size="<<sizeL<<endl;

	for (int npos1 = 0 ; npos1 <sizeL;npos1++) //translational
		for (int poss1 = 0; poss1 <sizeJ;poss1++) //rotational
		{
			int POS1=npos1*sizeJ+poss1;
			K1[POS1]=KK[npos1];
			L1[POS1]=LL[npos1];
			N1[POS1]=NN[npos1];
			ML1[POS1]=MML[npos1];
			J1[POS1]=JJ[poss1];
			MJ1[POS1]=MMJ[poss1];
		}

	cout<<" InitNumb done"<<endl;
}
void RotEnergy(int Jmax, int sizeJ, double B, double D, double *Hrot)
{
	cout<<" Rotational levelf of free HF:"<< endl;
	for (int i1=0;i1<sizeJ;i1++)
	{               Hrot[i1]=0.;}


	int pos1 =-1;
	for (int J=0;J<Jmax+1;J++)
		for (int M=-J;M<J+1;M++)
		{
			pos1+=1;
			Hrot[pos1]=B*(double)(J*(J+1))-D*(double)(J*J*(J+1)*(J+1));// in cm-1

			if (M==0)
				cout<<" J="<< J<<"  Energy="<<  Hrot[pos1]<<endl;

		} //loop over J,M
	cout<<" RotEnergy   done"<<endl;
}



void HbarEnergy(int *N1,int sizeL,int sizeJ,double hbar,double omega,double jpcm,double *Hbar)
{
	for (int pos = 0 ; pos < sizeL;pos++)
	{       
		double n=(double)N1[pos*sizeJ];
		Hbar[pos]=(hbar*omega*(1.5 + n) / jpcm ) ;
		cout<<" Hbar   pos "<<pos<<"  "<<Hbar[pos]<<endl;
	}
	cout<<" Hbar done"<<endl;
}


///Harmonic oscillator energies to subtract from V to get deltaV
//void HOEnergy(double dr,int Ngrid,int sizeL,int sizeJ,double omega,double massHF,double jpcm,int *K1,int *L1,int *ML1,double ***BasLL,double **HO)
void HOEnergy(int Ngrid,double *xyzgrid,double *xyzvec,int sizeL,int sizeJ, double omega,double massHF,double jpcm,int *K1,int *L1,int *ML1,double ***BasLL,double **HO)
{
	double ** norm; //normalization of the translational basis
	norm=doubleMatrix(sizeL,sizeL);

	for (int i=0;i<sizeL;i++)
		for (int j=0;j<i+1;j++)
		{
			HO[i][j] = 0.;
			double resul=0.;
			norm[i][j]=0.;
			for (int ix   = 0; ix   < Ngrid; ix++)
			{
	//			double xyzgrid =0.+ix*dr;//in nm
				resul     +=BasLL[i][j][ix]*xyzvec[ix]*xyzvec[ix]*xyzgrid[ix]*xyzgrid[ix]/pow(10.,18.)*delta(L1[i*sizeJ],L1[j*sizeJ])*delta(ML1[i*sizeJ],ML1[j*sizeJ]);//*xyzgrid*xyzgrid/pow(10.,18.);//*xyzgrid[ix]*xyzgrid[ix]/pow(10.,18.)
				norm[i][j]+=BasLL[i][j][ix]*delta(L1[i*sizeJ],L1[j*sizeJ])*delta(ML1[i*sizeJ],ML1[j*sizeJ]);
			}//grid over R
			HO[i][j]=resul*(0.5*massHF*omega*omega)/jpcm;
//			printf("i  %d  j  %d   K1 %d  K2 %d   HO: %0.16f    %0.16f\n", i,j,K1[i*sizeJ],K1[j*sizeJ],HO[i][j],norm[i][j]);
			HO[j][i]=HO[i][j];
		}

	//NOrmalisation
/*	for (int i=0;i<sizeL;i++)
		for (int j=0;j<i+1;j++)
			for (int ix   = 0; ix   < Ngrid; ix++)
				if (abs(norm[i][j])>pow(10.,16.))
					BasLL[i][j][ix] *=(1./norm[i][j]);
*/
}
void HOEnergyNew(int sizeL,int sizeJ,double omega,double massHF,double nu,double jpcm,int *K1,int *L1,int *ML1,double **HO)
{
        double ** norm; //normalization of the translational basis
        norm=doubleMatrix(sizeL,sizeL);

        for (int i=0;i<sizeL;i++)
                for (int j=0;j<sizeL;j++)
                {
                        HO[i][j] = 0.;
                        double resul=0.;
                        norm[i][j]=0.;
                }

        //NOrmalisation
/*      for (int i=0;i<sizeL;i++)
                for (int j=0;j<i+1;j++)
                        for (int ix   = 0; ix   < Ngrid; ix++)
                                if (abs(norm[i][j])>pow(10.,16.))
                                        BasLL[i][j][ix] *=(1./norm[i][j]);
*/


        for (int pos=0;pos<sizeL;pos++)
                for (int poss=0;poss<sizeL;poss++)
                        {

  long double Norm1=sqrt(sqrt(2.*nu*nu*nu/PI)*(pow(2.,(double)K1[pos*sizeJ]+2.*(double)L1[pos*sizeJ]+3.)*tgamma((double)(K1[pos*sizeJ]+1))*pow(nu,(double)L1[pos*sizeJ]))/doublefact(2*K1[pos*sizeJ]+2*L1[pos*sizeJ]+1));

  long double Norm2=sqrt(sqrt(2.*nu*nu*nu/PI)*(pow(2.,(double)K1[poss*sizeJ]+2.*(double)L1[poss*sizeJ]+3.)*tgamma((double)(K1[poss*sizeJ]+1))*pow(nu,(double)L1[poss*sizeJ]))/doublefact(2*K1[poss*sizeJ]+2*L1[poss*sizeJ]+1));

double m =(double)(L1[poss*sizeJ]+L1[pos*sizeJ]+1)/2.;
double m2=(double)((L1[poss*sizeJ]+L1[pos*sizeJ]+1+2)/2.);
int tau=K1[pos*sizeJ]-K1[poss*sizeJ];
long double tmp=0.;
long double tmp2=0.;
if (tau >= 0 )
{
for (int s=0;s<=K1[poss*sizeJ];s++)
{
tmp+=Binomial(m-(double)L1[pos*sizeJ]-0.5,(double)(tau+s))*Binomial((double)(m-L1[poss*sizeJ])-0.5,(double)s)*expl(lgammal((long double)(m+K1[poss*sizeJ]-s+1))-lgammal((long double)(K1[poss*sizeJ]-s+1)));
tmp2+=Binomial(m2-(double)L1[pos*sizeJ]-0.5,(double)(tau+s))*Binomial((double)(m2-L1[poss*sizeJ])-0.5,(double)s)*expl(lgammal((long double)(m2+K1[poss*sizeJ]-s+1))-lgammal((long double)(K1[poss*sizeJ]-s+1)));
}
}
else
{
tau*=-1.;
for (int s=0;s<=K1[pos*sizeJ];s++)
{
tmp+=Binomial(m-(double)L1[poss*sizeJ]-0.5,(double)(tau+s))*Binomial((double)(m-L1[pos*sizeJ])-0.5,(double)s)*expl(lgammal((long double)(m+K1[pos*sizeJ]-s+1))-lgammal((long double)(K1[pos*sizeJ]-s+1)));
tmp2+=Binomial(m2-(double)L1[poss*sizeJ]-0.5,(double)(tau+s))*Binomial((double)(m2-L1[pos*sizeJ])-0.5,(double)s)*expl(lgammal((long double)(m2+K1[pos*sizeJ]-s+1))-lgammal((long double)(K1[pos*sizeJ]-s+1)));
}
}

 norm[pos][poss]=tmp*pow(-1.,double(K1[pos*sizeJ]+K1[poss*sizeJ]))*delta(L1[pos*sizeJ],L1[poss*sizeJ])*delta(ML1[pos*sizeJ],ML1[poss*sizeJ])*Norm1*Norm2/(4.*nu*sqrt(pow(2.*nu,2.*m)));
 HO[pos][poss]=tmp2*pow(-1.,double(K1[pos*sizeJ]+K1[poss*sizeJ]))*delta(L1[pos*sizeJ],L1[poss*sizeJ])*delta(ML1[pos*sizeJ],ML1[poss*sizeJ])*Norm1*Norm2/(4.*nu*sqrt(pow(2.*nu,2*m2)))*(0.5*massHF*omega*omega)/jpcm;

           if(HO[pos][poss]!=0.)
cout << pos<<" "<<poss<<"  norm :  "<<norm[pos][poss]<<"  HO : "<<HO[pos][poss]<<"  |  K "<<K1[pos*sizeJ]<<" L "<<L1[pos*sizeJ]<<  " K' "<<K1[poss*sizeJ]<<" L' "<<L1[poss*sizeJ]<<endl;

                        }

}

//Calculating number of expansion coeff (degenerate MJ , ML)
void NumbExpanCoef(int npot,int *pV1,int *pV2, int *pV, int *mp, int &nV)
{
	pV1[0]=0;
	pV1[1]=0;
	pV1[2]=0;
	pV1[3]=1;
	pV1[4]=1;
	pV1[5]=1;
	pV1[6]=2;
	pV1[7]=2;
	pV2[0]=0;
	pV2[1]=6;
	pV2[2]=6;
	pV2[3]=1;
	pV2[4]=5;
	pV2[5]=5;
	pV2[6]=2;
	pV2[7]=4;
	pV[0]=0;
	pV[1]=6;
	pV[2]=6;
	pV[3]=0;
	pV[4]=6;
	pV[5]=6;
	pV[6]=0;
	pV[7]=6;
	mp[0]=0;
	mp[1]=0;
	mp[2]=5;
	mp[3]=0;
	mp[4]=0;
	mp[5]=5;
	mp[6]=0;
	mp[7]=0;
	nV=0;
	for (int i = 0;i<npot;i++) ///NPOT
		for (int r1=-pV1[i];r1<=pV1[i];r1++)
			for (int r2=-pV2[i];r2<=pV2[i];r2++)
				if ((abs(pV1[i]-pV2[i])>pV[i]) || (pV[i] > (pV1[i]+pV2[i])) || ((r1+r2+mp[i]) !=0)  )
				{
					nV +=0;
				}
				else
				{
					nV+=1;

				}    
}

//void CalcCoef(double dr,int Ngrid,int npot,int nV,int *pV1,int *pV2,int *pV,int *mp,int *VL,int *VML,int *VJ,int *VMJ,int *VS,int *VMS,double **coef)
void CalcCoef(int Ngrid,double *xyzgrid,int npot,int nV,int *pV1,int *pV2,int *pV,int *mp,int *VL,int *VML,int *VJ,int *VMJ,int *VS,int *VMS,double **coef)
{
	double kcal2cm=4.184*83.59539;
	double **x;
	x=doubleMatrix(npot,Ngrid);

	for (int ix=0;ix<Ngrid;ix++)
	{  //double xyzgrid =0.+ix*dr;
		double R=xyzgrid[ix]*10.;
		double R2=R*R;
		double R3=R2*R;
		double R4=R2*R2;
		
		   x[0][ix]=11.6480952*R4+7.011576190*R2-13.7432 ;
		   x[1][ix]=0.;//0.07610476*R4+0.002865809*R2+0.024668;
		   x[2][ix]=0.;//0.185607142*R4+0.104138214*R2-0.0448  ;
		   x[3][ix]=0.;//1.93809523*R3-0.972523809*R          ;
		   x[4][ix]=0.;//-0.48571428*R3-0.108571428*R          ;
		   x[5][ix]=0.;//0.148095238*R3+0.170576190*R          ;
		   x[6][ix]=0.;//0.889595238*R4+0.107641190*R2         ;
		   x[7][ix]=-0.30880952*R4+0.286602381*R2;

		 
		//              REAL COEF
/*
		x[0][ix]=11.6480952*R4+7.011576190*R2-13.7432 ;
		x[1][ix]=0.07610476*R4+0.002865809*R2+0.024668;
		x[2][ix]=0.185607142*R4+0.104138214*R2-0.0448  ;
		x[3][ix]=1.93809523*R3-0.972523809*R          ;
		x[4][ix]=-0.48571428*R3-0.108571428*R          ;
		x[5][ix]=0.148095238*R3+0.170576190*R          ;
		x[6][ix]=0.889595238*R4+0.107641190*R2         ;
		x[7][ix]=-0.30880952*R4+0.286602381*R2;
*/
		//              printf("A0000  %0.12f \n",x[0][ix]);
	}
	int ic=-1;
	for (int i = 0;i<npot;i++) ///NPOT
		for (int r1=-pV1[i];r1<=pV1[i];r1++)
			for (int r2=-pV2[i];r2<=pV2[i];r2++)
			{
				if ((abs(pV1[i]-pV2[i])>pV[i]) || (pV[i] > (pV1[i]+pV2[i])) || ((r1+r2+mp[i]) !=0)  )
				{
					ic +=0;
				}
				else
				{
					ic+=1;
					VL[ic]=pV1[i];
					VML[ic]=r1;
					VJ[ic]=pV2[i];
					VMJ[ic]=r2;
					VS[ic]=pV[i];
					VMS[ic]=mp[i];


					for (int ix=0;ix<Ngrid;ix++)
						coef[ic][ix]=x[i][ix]*W3j(pV1[i],pV2[i],pV[i],r1,r2,mp[i])*kcal2cm*pow(-1.,(double)(r1+r2))/sqrt((2.0*(double)pV1[i]+1.0)*(2.0*(double)pV2[i]+1.0));

				}
			}



}


void CalcCoefNew(int npot,int nV,int sizeL, int sizeJ,double nu,int *K1,int *L1,int *ML1,int *pV1,int *pV2,int *pV,int *mp,int *VL,int *VML,int *VJ,int *VMJ,int *VS,int *VMS,double ***V)
{
        long double kcal2cm=4.184*83.59539;
        double **x;
        int maxpow=5;
        x=doubleMatrix(npot,maxpow);

                   x[0][0]=-13.7432 ;
                   x[0][1]=0.;
                   x[0][2]=7.011576190 ;
                   x[0][3]=0.;
                   x[0][4]=11.6480952 ;


                   x[1][0]=0.024668;
                   x[1][1]=0.;
                   x[1][2]=0.002865809;
                   x[1][3]=0.;
                   x[1][4]=0.07610476;

                   x[2][0]=-0.0448  ;
                   x[2][1]=0.;
                   x[2][2]=0.104138214 ;
                   x[2][3]=0.;
                   x[2][4]=0.185607142  ;

                   x[3][0]=0.;
                   x[3][1]=-0.972523809  ;
                   x[3][2]=0.;
                   x[3][3]=1.93809523  ;
                   x[3][4]=0.;

                   x[4][0]=0.;
                   x[4][1]=-0.108571428;
                   x[4][2]=0.;
                   x[4][3]=-0.48571428 ;
                   x[4][4]=0.;

                   x[5][0]=0.;
                   x[5][1]=0.170576190 ;
                   x[5][2]=0.;
                   x[5][3]=0.148095238 ;
                   x[5][4]=0.;


                   x[7][0]=0.;
                   x[7][1]=0.;
                   x[7][2]=0.286602381;
                   x[7][3]=0.;
                   x[7][4]=-0.30880952;


                   x[6][0]=0.;
                   x[6][1]=0.;
                   x[6][2]=0.107641190;
                   x[6][3]=0.;
                   x[6][4]=0.889595238;



        for (int i=0;i<maxpow;i++)
        { 
//                   x[0][i]=0.;
//                   x[1][i]=0.; 
//                   x[2][i]=0.;
//                   x[3][i]=0.;
//                   x[4][i]=0.;
//                   x[5][i]=0.;
//                   x[6][i]=0.;
//                   x[7][i]=0.;
        }

        for (int pos=0;pos<sizeL;pos++)
                for (int poss=0;poss<sizeL;poss++)
{                     
 long double Norm1=sqrtl(sqrtl((long double)(2.*nu*nu*nu/PI))*(powl((long double)2,(long double)(K1[pos*sizeJ]+2*L1[pos*sizeJ]+3))*tgammal((long double)(K1[pos*sizeJ]+1))*powl((long double)nu,(long double)L1[pos*sizeJ]))/(long double)doublefact(2*K1[pos*sizeJ]+2*L1[pos*sizeJ]+1));

 long double Norm2=sqrtl(sqrtl((long double)(2.*nu*nu*nu/PI))*(powl((long double)2,(long double)(K1[poss*sizeJ]+2*L1[poss*sizeJ]+3))*tgammal((long double)(K1[poss*sizeJ]+1))*powl((long double)nu,(long double)L1[poss*sizeJ]))/(long double)doublefact(2*K1[poss*sizeJ]+2*L1[poss*sizeJ]+1));

        int ic=-1;
        for (int i = 0;i<npot;i++)
                for (int r1=-pV1[i];r1<=pV1[i];r1++)
                        for (int r2=-pV2[i];r2<=pV2[i];r2++)
                        {
                                if ((abs(pV1[i]-pV2[i])>pV[i]) || (pV[i] > (pV1[i]+pV2[i])) || ((r1+r2+mp[i]) !=0)  )
                                {
                                        ic +=0;
                                }
                                else
                                {
                                        ic+=1;
                                        VL[ic]=pV1[i];
                                        VML[ic]=r1;
                                        VJ[ic]=pV2[i];
                                        VMJ[ic]=r2;
                                        VS[ic]=pV[i];
                                        VMS[ic]=mp[i];
                                


V[ic][pos][poss]=0.;
for (int ix=0;ix<maxpow;ix++)
{
double m =(double)(L1[poss*sizeJ]+L1[pos*sizeJ]+1+ix)*0.5;
int tau=K1[pos*sizeJ]-K1[poss*sizeJ];
long double tmp=0.;


if (tau >= 0. )
{
for (int s=0;s<=K1[poss*sizeJ];s++)
{
tmp+=Binomial(m-0.5-(double)L1[pos*sizeJ],(double)(tau+s))*Binomial(m-0.5-(double)L1[poss*sizeJ],(double)s)*expl(lgammal((long double)(m+K1[poss*sizeJ]-s+1))-lgammal((long double)(K1[poss*sizeJ]-s+1)));
}
}
else
{
tau*=-1.;
for (int s=0;s<=K1[pos*sizeJ];s++)
{
tmp+=Binomial(m-(double)L1[poss*sizeJ]-0.5,(double)(tau+s))*Binomial((double)(m-L1[pos*sizeJ])-0.5,(double)s)*expl(lgammal((long double)(m+K1[pos*sizeJ]-s+1))-lgammal((long double)(K1[pos*sizeJ]-s+1)));
}
}


 V[ic][pos][poss]+=tmp*pow(-1.,(double)(K1[pos*sizeJ]+K1[poss*sizeJ]))*Norm1*Norm2/((long double)(4.*nu)*sqrtl(powl((long double)(2.*nu),(long double)(2.*m))))*x[i][ix]*W3j(pV1[i],pV2[i],pV[i],r1,r2,mp[i])*kcal2cm*pow(-1.,(double)(r1+r2))/sqrt((double)((2*pV1[i]+1)*(2*pV2[i]+1)))*pow(pow(10.,10.),(double)(ix));//*delta(L1[pos*sizeJ],L1[poss*sizeJ])*delta(ML1[pos*sizeJ],ML1[poss*sizeJ]);

//if (i==0)
//if (pos==4 && poss==0 && ic==0)
//cout<<" TMP :"<<tmp<<" x "<<x[i][ix]<<" W3j "<<W3j(pV1[i],pV2[i],pV[i],r1,r2,mp[i])<<" V "<<V[ic][pos][poss]<<endl;


}
//cout <<"ic "<<ic<<" i "<<pos<<" j "<<poss<<"    V[ic]ij = "<<V[ic][pos][poss]<<"  |  K "<<K1[pos*sizeJ]<<" L "<<L1[pos*sizeJ]<<  " K' "<<K1[poss*sizeJ]<<" L' "<<L1[poss*sizeJ]<<endl;
}
}
}
}


long double Binomial(double a,double b)
{
if (a<0 || b<0 )
{return 0.;}

if (b > a)
{return 0.;}
else
{long double tmp=expl(lgammal((long double)(a+1))-lgammal((long double)(a-b+1))-lgammal((long double)(b+1)));
return tmp;}
}

void InitBasisNew(int Kmax, int Lmax, int sizeL, long double nu, double **BasL)
{
        int K[sizeL];
        int L[sizeL];
        int ML[sizeL];

        int ps=-1;
        for (int K1=0;K1<Kmax+1;K1++)
                for (int L1=0;L1<Lmax+1;L1++)
                        for (int ML1=-L1;ML1<=L1;ML1++)
                        {
                                ps+=1;
                                K[ps]=K1;
                                L[ps]=L1;
                                ML[ps]=ML1;
                        }


        for (int pos=0;pos<sizeL;pos++)
                for (int poss=0;poss<sizeL;poss++)
                        {

                               long double Norm1=sqrt(sqrt(2.*nu*nu*nu/PI)*(pow(2.,(double)K[pos]+2.*(double)L[pos]+3.)*tgamma((double)(K[pos]+1))*pow(nu,(double)L[pos]))/doublefact(2*K[pos]+2*L[pos]+1));

                               long double Norm2=sqrt(sqrt(2.*nu*nu*nu/PI)*(pow(2.,(double)K[poss]+2.*(double)L[poss]+3.)*tgamma((double)(K[poss]+1))*pow(nu,(double)L[poss]))/doublefact(2*K[poss]+2*L[poss]+1));

long double m =(long double)((L[poss]+L[pos]+1)*0.5);
int tau=K[pos]-K[poss];
long double tmp=0.;
if (tau >= 0 )
{
for (int s=0;s<=K[poss];s++)
{
tmp+=Binomial(m-(double)L[pos]-0.5,(double)(tau+s))*Binomial((double)(m-L[poss])-0.5,(double)s)*expl(lgammal((long double)(m+K[poss]-s+1))-lgammal((long double)(K[poss]-s+1)));
}
}
else
{
tau*=-1.;
for (int s=0;s<=K[pos];s++)
{
tmp+=Binomial(m-(double)L[pos]-0.5,(double)(tau+s))*Binomial((double)(m-L[poss])-0.5,(double)s)*expl(lgammal((long double)(m+K[pos]-s+1))-lgammal((long double)(K[pos]-s+1)));
}
}



 BasL[pos][poss]=tmp*pow(-1.,(double)(K[pos]+K[poss]))*(double)(delta(L[pos],L[poss])*delta(ML[pos],ML[poss])*Norm1*Norm2)/((4.*nu)*sqrtl(powl((long double)(2.*nu),(long double)(L[pos]+L[poss]+1))));
//if (pos==poss && ML[pos]==0 && ML[poss]==0)
//cout <<" BASL    "<<pos<<" "<<poss<<" "<<BasL[pos][poss]<<" K "<<K[pos]<<" L "<<L[pos]<<  " K' "<<K[poss]<<" L' "<<L[poss]<<endl;
                        }
}
