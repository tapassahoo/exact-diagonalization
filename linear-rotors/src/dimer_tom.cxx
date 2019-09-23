#include "BF.h"
#include "random.h"
/* 
CHOOSING BASIS == 0 
 This program will calculate the energy levels of a He-He dimer system using the Colbert-Miller basis found in JCP 96,1982.
 In this case, we divide our gridpoints into equal intervals
 
 CHOOSING BASIS == 1
 This program will calculate the energy levels of a He-He dimer system using the Tridiagonal-Morse basis found in JCP 97,5. 
 
 */

// FUNCTIONS
matrix kinetic(int i, double mu, double delta_r);  // for Colbert-Miller only
matrix rotation(int size, int J, double mu, double delta_r, vector gridR);
diagmat potential(int size, double delta_r, vector gridR,int system);
matrix tridiagKE(int size, double mu, vector &gridR, vector &weightR);
double Aziz(double r);
double buckpigs(double r);
double Silvera(double r);
double SilveraFixed(double r);
double Buck(double r);
double HeH2(double r);
void ftridiagmorse(int sizeR,matrix &KR,vector &weightR,vector &gridR,double massHe);
vector tridiagMorse(int sizeR,double De,double mu,double frr,matrix &z,matrix &p2,double alpha, double gamma, int Ne);
void addtobasis(vector &v,matrix &B,int n,int col);


// CONSTANTS
static const double BohrToA = 0.529177249; // Bohr to angstrom
static const double au_to_amu = 5.4857989586762187e-4; // au to amu
static const double HtoW = 2.19474631371017e5;  // hartree to cm^-1
//static const double  hatocm=219474.629;
static const double hartree_to_joule = 4.359748e-18;
static const double h_bar = 1.0;
//static const double MofH2 = 2.01612085117; // mass of H2 in amu
static const double MofH2 = 2.015650642; // nist value
static const double MofD2 = 2.0141017780*2.;// nist
static const double MofT2 = 4.0*2.;// nist
static const double MofHe4 = 4.0026032497; // nist mass of He in amu
static const double kB = 3.1668153e-6; // Hartree/Kelvin

// MAIN PROGRAM
int main(int argc, char **argv)
{
  if(argc != 5)
    {
      std::cerr << "<R_min> <Max R (Angstroms)>, <# of gridpoints N>, <Max J>"<<std::endl;
		exit(0);
    }
  
  // COMMAND PROMPT INPUT
  // get range of r (0,max_value)
  double R_min_bohr = atof(argv[1])*atob;
  double R_max_bohr = atof(argv[2])*atob;
  // get # of gridpoints N
  int gridpoints = atoi(argv[3]);
  // get max value of J
  int J = atoi(argv[4]);
  
  // CONSTANTS PARTICULAR TO SYSTEM
  std::cout<<" mass of H2 = "<<MofH2<<std::endl;
  double mu = MofH2 /2./au_to_amu; 
  
  double r = R_max_bohr-R_min_bohr;  // max value for r
  double delta_r = r/(double)gridpoints; // for Colbert-Miller
	
  // the matrix elements to create H
  matrix KE(gridpoints,gridpoints);
  matrix RO(gridpoints,gridpoints);
  diagmat PE0(gridpoints);
  diagmat PE0sym(gridpoints);
  diagmat PE1(gridpoints);
  diagmat PE1sym(gridpoints);
  vector gridR(gridpoints);
  vector weightR(gridpoints);
  matrix H0(gridpoints,gridpoints);
  matrix H0sym(gridpoints,gridpoints);
  matrix H1(gridpoints,gridpoints);
  matrix H1sym(gridpoints,gridpoints);
  vector energies(gridpoints);
  matrix densitymatrix(gridpoints,gridpoints);
  vector densitylanczos(gridpoints);
  double Z=0.;  // partition function - declaration to 0 at initialization
  double area=0;
  for(int i=0;i<gridpoints;i++)
    gridR(i)=R_min_bohr+((double)(i)*delta_r);
  
  double k=sqrt(2.*mu*(9.e-4/HtoW));
 
  std::ifstream potn0("nu0dense");
  std::ifstream potn1("nu1dense");
  int npoints=195;
  vector rfromfile(npoints);
  vector potfromfile0(npoints);
  vector potfromfile0sym(npoints);
  vector potfromfile1(npoints);
  vector potfromfile1sym(npoints);
  double dummy;

  ofstream potout0("V0.dat");
  ofstream potout0sym("V0sym.dat");
  ofstream potout1("V1.dat");
  ofstream potout1sym("V1sym.dat");
  ofstream potout1diff("V1diff.dat");
  ofstream potout0diff("V0diff.dat");
  
  for (int i=0;i<npoints;i++){
    potn0>>rfromfile(i)>>potfromfile0(i)>>potfromfile0sym(i);
    potn1>>rfromfile(i)>>potfromfile1(i)>>dummy>>potfromfile1sym(i)>>dummy;
    
    potout0<<rfromfile(i)*atob<<" "<<potfromfile0(i)<<endl;
    potout0sym<<rfromfile(i)*atob<<" "<<potfromfile0sym(i)<<endl;
    potout1<<rfromfile(i)*atob<<" "<<potfromfile1(i)<<endl;
    potout1sym<<rfromfile(i)*atob<<" "<<potfromfile1sym(i)<<endl;
    potout1diff<<rfromfile(i)*atob<<" "<<potfromfile1(i)-potfromfile1sym(i)<<endl;
  }
  
  rfromfile=atob*rfromfile;
  
  Interp V0_func(npoints,rfromfile,potfromfile0);
  Interp V0_func_sym(npoints,rfromfile,potfromfile0sym);
  Interp V1_func(npoints,rfromfile,potfromfile1);
  Interp V1_func_sym(npoints,rfromfile,potfromfile1sym);

  KE = kinetic(gridpoints,mu,delta_r);
  RO = rotation(gridpoints, J, mu, delta_r, gridR);
  for (int i=0;i<gridpoints;i++){
  	PE0(i)= V0_func.interp(gridR(i))/hatocm;   	 	  
  	PE0sym(i)= V0_func_sym.interp(gridR(i))/hatocm;   	 	  
  	PE1(i)= V1_func.interp(gridR(i))/hatocm;   	 	  
  	PE1sym(i)= V1_func_sym.interp(gridR(i))/hatocm;   	 	  
   }
  
  if (J >0) {
    H0 = KE+RO+PE0;
    H0sym = KE+RO+PE0sym;
    H1 = KE+RO+PE1;
    H1sym = KE+RO+PE1sym;
  }
  else {
    H0 = KE+PE0;
    H0sym = KE+PE0sym;
    H1 = KE+PE1;
    H1sym = KE+PE1sym;
  }

  vector energies0= diag(H0); // diagonalizing the matrix actually modifies the matrix      
  vector energies0sym= diag(H0sym); // diagonalizing the matrix actually modifies the matrix      
  vector energies1 = diag(H1); // diagonalizing the matrix actually modifies the matrix      
  vector energies1sym = diag(H1sym); // diagonalizing the matrix actually modifies the matrix      


  std::cout.precision(5);
  std::cout<<"E_0 v=0       = "<<energies0(0)*hatocm<<" cm-1"<<std::endl;
  std::cout<<"E_0 (sym) v=0 = "<< energies0sym(0)*hatocm<<" cm-1"<<std::endl;
  std::cout<<"E_0 v=1       = "<< energies1(0)*hatocm<<" cm-1"<<std::endl;
  std::cout<<"E_0 (sym) v=1 = "<< energies1sym(0)*hatocm<<" cm-1"<<std::endl;
  std::cout<<endl;
  std::cout.precision(4);
  std::cout<<"shift=          "<<(energies1(0)-energies0(0))*hatocm<<" cm-1"<<std::endl;
  std::cout<<"shift (PI sym)= "<<(energies1sym(0)-energies0(0))*hatocm<<" cm-1"<<std::endl;
  std::cout<<"shift (PI sym sym)= "<<(energies1sym(0)-energies0sym(0))*hatocm<<" cm-1"<<std::endl;
  std::ofstream eigenvalues("energy.out");
  for(int n=0; n<gridpoints; n++){
    eigenvalues << energies0(n)*hatocm<<" "<<energies0sym(n)*hatocm<<" "<<energies1(n)*hatocm<<" "<<energies1sym(n)*hatocm<<std::endl;
  }

  
  std::ofstream rhoout("rho0");
    std::cout<<endl;


  vector rho0(gridpoints);
  vector rho0sym(gridpoints);

  // PT1 shifts and psi^2 output
  double PTshift=0.;
  double PTshift_sym=0.;
  double PTshift_sym_sym=0.;
  for(int i=0; i<gridpoints; i++)  {
    rho0(i)=fabs(pow(H0(i,0)/sqrt(delta_r/atob),2.)); //unit jacobian psi(r)=r phi(r)
    rho0sym(i)=fabs(pow(H0sym(i,0)/sqrt(delta_r/atob),2.)); //unit jacobian psi(r)=r phi(r)
    rhoout << gridR(i)/atob << " " << rho0(i)<<" "<< rho0sym(i)<<std::endl;
    PTshift+=pow(H0(i,0),2.)*(PE1(i)-PE0(i));
    PTshift_sym+=pow(H0sym(i,0),2.)*(PE1sym(i)-PE0(i));
    PTshift_sym_sym+=pow(H0(i,0),2.)*(PE1sym(i)-PE0sym(i));
  }
  std::cout<<"PT shift =        "<<PTshift*hatocm<<" cm-1"<<std::endl;
  std::cout<<"PT shift (PI sym) = "<<PTshift_sym*hatocm<<" cm-1"<<std::endl;
  std::cout<<"PT shift (PI sym sym) = "<<PTshift_sym*hatocm<<" cm-1"<<std::endl;

}

matrix kinetic(int size, double mu, double delta_r)
{
  matrix T(size,size);
  
  int i,ip;
  
  // do a for loop for both i's
  for(i=0; i<size; i++)
    {
      double ivalue=(double)i+1.;
      for(ip=0; ip<size; ip++)
	{	  	  
	  double ipvalue=(double)ip+1.;
	  double T1 = h_bar*h_bar / (2.0*mu*delta_r*delta_r) * pow(-1.,ivalue-ipvalue);

	  if(i==ip)
	    T(i,ip) = T1 * ((M_PI*M_PI/3.0)-(1./(2.0*ivalue*ivalue)));
	    //T(i,ip) =  (M_PI*M_PI/(mu*delta_r*delta_r))*((double)size*(double)size+2.)/6.;
	  else
	    T(i,ip) = T1 * ((2.0/((ivalue-ipvalue)*(ivalue-ipvalue)))-(2.0/((ivalue+ipvalue)*(ivalue+ipvalue))));
	  //T(i,ip) =  pow(-1.,(double)(ivalue-ipvalue))*(M_PI*M_PI/(mu*delta_r*delta_r))/pow(sin((double)(ivalue-ipvalue)*M_PI/(double)size),2.);
	}
    }
  
  //  std::cout << "KINETIC MATRIX" << std::endl;  
  //   for(i=1; i<size; i++)
  //     {
  //       for(ip=1; ip<size; ip++)
  // 	{	  
  // 	  std::cout.width(10);
  // 	  std::cout.precision(4);
  
  // // 	  std::cout << T(i-1,ip-1) << " ";
  // 	}
  // //       std::cout << std::endl;
  //     }
  
  return T;
  
}

matrix rotation(int size, int J, double mu, double delta_r, vector gridR)
{
  matrix Rot(size,size);
  int i,ip;
  
  // use this method if using Colbert-Miller

  for(i=0; i<size; i++)
    {
      double ivalue = (double)i+1.;
      for(ip=0; ip<size; ip++)
	{
	  double ipvalue = (double)ip+1.;
	  if(i==ip)
	    Rot(i,ip) = (h_bar*h_bar/(2.0*mu*gridR(i)*gridR(i)) ) *(double)(J+1)*(double)J;
	  else Rot(i,ip) =0.;
	}
    }      
  
  
//   std::cout << "ROTATIONAL MATRIX" << std::endl;

//   for(i=1; i<size; i++)
//     {
//       for(ip=1; ip<size; ip++)
// 	{
	  
// 	  std::cout.width(10);
// 	  std::cout.precision(4);
	  
// // 	  std::cout << Rot(i-1,ip-1) << " ";
// 	}
//  //       std::cout << std::endl;
//     }
  
  return Rot;
}

diagmat potential(int size, double delta_r, vector gridR, int system) // THIS MUST BE DIAGONAL!!!!
{
  diagmat Pot(size);
  int i,ip;


  for(i=0; i<size; i++)	{
	 
	if (system==0) Pot(i) = Aziz(gridR(i));
	if (system==1) Pot(i) = Silvera(gridR(i));
	//if (system==1) Pot(i) = SilveraFixed(gridR(i));
	if (system==2) Pot(i) = Buck(gridR(i));
	if (system==3) Pot(i) = Silvera(gridR(i));
	if (system==4) Pot(i) = Buck(gridR(i));
	if (system==6) Pot(i) = Buck(gridR(i));
	if (system==5) Pot(i) = HeH2(gridR(i));
	
  }  
  return Pot;
}


// 
matrix tridiagKE(int size, double mu, vector &gridR, vector &weightR)
{

  // STUFF FOR TRIDIAGONAL MORSE
  // temp storage location for these two variables
  matrix KR(size,size);

  //  vector gridR(size);
  
  ftridiagmorse(size, KR, weightR, gridR, mu);

  return KR;
}

double Buck( double r)
{
	double V;
	double a=101.4;
	double beta = 2.779;
	double alpha= 0.08;
	double c6=7.254;
	double c8=36.008;
	double c10=225.56;
	double d=5.102;
	//r=sqrt(x);
	r/=atob;
	double epsilon = 1.;
	if (r <= d) epsilon = exp(-1*pow((d/r-1),2));
	//potl [lrun] = 11604.45*(a*exp((-1.*beta*r)-(alpha*pow(r,2))) - epsilon*((c6/pow(r,6)) + (c8/pow(r,8)) + (c10/pow(r,10))));
	V = (1./27.2113845)*(a*exp((-1.*beta*r)-(alpha*pow(r,2))) - epsilon*((c6/pow(r,6)) + (c8/pow(r,8)) + (c10/pow(r,10))));
	//V=buckpigs(r*atob); //test
	return V;
}
double buckpigs(double r){
  double V,epsilon;
  double a=101.4;
  double beta = 2.779;
  double alpha=0.08;
  double c6=7.254;
  double c8=36.008;
  double c10=225.56;
  double d=5.102;
  r/=atob;
  epsilon = 1.;
  if (r <= d) epsilon = exp(-1*pow((d/r-1),2));
  V = kB*11604.505*(a*exp((-1.*beta*r)-(alpha*pow(r,2))) - epsilon*((c6/pow(r,6)) + (c8/pow(r,8)) + (c10/pow(r,10))));
  return V;
}
double Silvera( double r)
{
  double f,a2i,a6i,rtm,V;
  double a=1.713;
  double alpha=1.567;
  double beta=-0.00993;
  double c6=12.14;
  double c8=215.2;
  double c9=-143.1;
  double c10=4813.9;
  double rm=3.41;
  rm*=atob;
  double d=1.28*rm;
  a2i=1./(r*r);
  a6i=a2i*a2i*a2i;
  rtm=((d/r)-1.0);
  f = (r < d) ? exp(-rtm*rtm) : 1.;
  V = (exp(a-alpha*r+beta*r*r)-f*a6i*(c6+a2i*(c8+c9/r+a2i*c10)));
  return V;
}
double SilveraFixed( double r)
{
  double f,a2i,a6i,rtm,V;
  double a=1.713;
  double alpha=1.5671;
  double beta=-0.00993;
  double c6=12.14;
  double c8=215.2;
  double c9=-143.1;
  double c10=4813.9;
  double rm=3.41;
  rm*=atob;
  double d=1.28*rm;
  a2i=1./(r*r);
  a6i=a2i*a2i*a2i;
  rtm=((d/r)-1.0);
  f = (r < d) ? exp(-rtm*rtm) : 1.;
   V = (exp(a-alpha*r+beta*r*r)-f*a6i*(c6+a2i*(c8+c9/r+a2i*c10)));
  return V;
}


double Aziz(double r)
{
  // parameters for the LM2M2 potential of Aziz JCP, 94, 8047 (1991)
  double AStar=1.89635353e5;
  double alphaStar=10.70203539;
  double c6=1.34687065;
  double c8=0.41308398;
  double c10=0.17060159;
  double C6=1.461; // au
  double C8=14.11; // au
  double C10=183.5; // au
  double betaStar=-1.90740649;
  double beta=-0.21631; // A^-2
  double D=1.4088;
  double epsilonoverk=10.97; // K
  double epsilon=epsilonoverk*kB;
  double rm=2.9695; // A
  rm*=atob;
  double sigma= 2.6417278; // A
  // add-on portions
  double Aa=0.0026;
  double x1=1.003535949;
  double x2=1.454790369;

  // HFD potential
  // aziz and chen, JCP, 67, 5719 (1977)
  // V(x) = epsilon V*(x)
  // V*(x) = A exp(-alpha x) - (C6/x^6+C8/x^8+C10/x^10) F(x)
  // F(x) = exp(-(D/x-1)^2) for x<D and F(x)=1 for >=D


  double x=r/rm;
  double B=2.*M_PI/(x2-x1);
  double VaStar=Aa*(sin(B*(x-x1)-M_PI/2.)+1.);
  if (x <x1 || x > x2) VaStar=0.;

  double F= exp(-pow((D/x-1.),2.));
  if (x >= D) F=1.;

  
  double VbStar=AStar*exp(-alphaStar*x+betaStar*x*x)-(c6/pow(x,6.)+c8/pow(x,8.)+c10/pow(x,10.))*F;
  
  double V=epsilon*(VaStar+VbStar);
  return V;
}
double HeH2(double r) 
{
	double V;
	double epsilon=4.189721e-05 ;
	double sigma=5.671145 ;
	double rc=8.35110 ;
	double c6=4.018 ; // units
	double c8=55.69 ;
	double c10=1031.0 ;		
	if (r <= rc) {
	double a6i=pow(sigma,6)/(r*r*r*r*r*r);
	V = 4*epsilon*a6i*(a6i-1);
	}
	else{
	double r3i=1.0/(r*r*r);
	double r4i=1.0/(r*r*r*r);
	V = (-1*(c6*r3i*r3i)-1*(c8*r4i*r4i)-1*(c10*r3i*r3i*r4i));
	}
	return V;
}

// Tridiagonal Morse Basis code
void ftridiagmorse(int sizeR,matrix &KR,vector &weightR,vector &gridR,double massHe)
{
	int i,j;
	double dummy;
	matrix z1(sizeR,sizeR);
	//  double De=120.;  // ORIGINAL VALUE ************************************
	double epsilonoverk=10.97; // K
	double epsilon=epsilonoverk*kB;
	//   double De=160.*epsilon;
	//  double De=1.*epsilon;
	double De=2.1*epsilon;
	std::cout << "De" << De << std::endl;
	
	//  double frr=.000005; ****************************************
	double rminAziz=2.9695*atob;
	double rminSilvera=3.41*atob;
	double rminBuck=5.102*atob;
	double deltar=.01;
	//double frr= .00001*(Silvera(rminSilvera+2.*deltar)-2.*Silvera(rminSilvera)-Silvera(rminSilvera-2.*deltar))/(4.*deltar*deltar);
	//double frr= .00001*(Buck(rminBuck+2.*deltar)-2.*Buck(rminBuck)-Buck(rminBuck-2.*deltar))/(4.*deltar*deltar);
	double frr= .000001*(Aziz(rminAziz+2.*deltar)-2.*Aziz(rminAziz)-Aziz(rminAziz-2.*deltar))/(4.*deltar*deltar);
	std::cout << "frr " << frr << std::endl;
	int Ne; 
	double gamma,alphaR;
	/* determination of the number of bound states (Ne +1) */
	Ne=(int)floor(2.*De/sqrt(frr/massHe)-1.);
	std::cout<<"Ne= "<<Ne<<std::endl;
	if (Ne > (sizeR-1)) Ne=sizeR-1;
	/* determination of gamma */
	gamma=(2.*De/sqrt(frr/massHe)-((double)Ne+1.));
	//gamma=1.;
	std::cout<<"gamma= "<<gamma<<std::endl;
	/*determination of alphaR*/
	alphaR=sqrt(frr/2./De);
	// std::cout<<gamma<<std::endl; 
	//gridz contain Gaussian quadrature points in z
	vector gridz1=tridiagMorse(sizeR,De,massHe,frr,z1,KR,alphaR,gamma,Ne);  // tridiagMorse returns the gridpoints wrt z
																			//gridR has it in r
																			//find out the normalization factor of the zero order polynomial
	double norm0=sqrt(alphaR/exp(lgamma(2.*gamma+1.))); // normalization for z
														// std::cout<<"norm is: "<<norm0<<std::endl;
														//find the weights in DVR of R
	double zvalue;
	// std::cout<<"alpha: "<<alphaR<<std::endl;
	//remorse is rmin,so r starts at zero
	double remorse=log(gridz1(sizeR-1)*alphaR/(2.*sqrt(2.*massHe*De)))/alphaR;
	std::cout<<"remorse= "<<remorse<<std::endl;
	//  double refactor=2.; // CHANGE THE STARTING POINT HERE *****************************
	double refactor=1.;
	//now r starts at refactor
	remorse+=refactor;
	//convert z to r
	for (i=0;i<sizeR;i++)
		gridR(i)=remorse-(log(gridz1(i)*alphaR/(2.*sqrt(2.*massHe*De)))/alphaR);
	//find the weightR1
	for (i=0;i<sizeR;i++){
		//dummy is now the value of the basis w/ poly,norm^.5, wt function^.5
		dummy=norm0*pow(gridz1(i),gamma+.5)*exp(-gridz1(i)/2.);
		//weightR=sqrt(weight/weightfunction)
		weightR(i)=dummy/z1(0,i);
		//if (dummy <= 1.e-300) weightR(i)=0.; /// please check
	}
	// output grid and vectors for R
	std::ofstream gridR1out("gR1");
	for (i=0;i<sizeR;i++) {  // WHERE ALL THE STUFF IS OUTPUTTED******* SEE LIMIT OF 3/4 ... CHI_FACTOR/z
							 // note that lgamma is the log of gamma function, so exp(lgamma)=gamma function
							 // the limit of the 3rd / 4th columns tends to -'ve infinity.
		gridR1out<<gridR(i)<<" "<<gridz1(i)<<" "<<weightR(i)<<" "<<sqrt(alphaR/exp(lgamma(2.*gamma+1.)))*pow(gridz1(i),gamma+.5)*exp(-gridz1(i)/2.)<<" "<<z1(0,i);
		
		// testing!
		//std::cout << "stuff " << weightR(i)/(sqrt(alphaR/exp(lgamma(2.*gamma+1.)))*pow(gridz1(i),gamma+.5)*exp(-gridz1(i)/2.) ) << std::endl;
		for (j=0;j<sizeR;j++){
			//      gridR1out<<pow(z1(j,i)/weightR(i),2.)<<" ";
		}
		gridR1out<<std::endl;
	}
	
	std::cout << "lgamma " << lgamma(1.+1.) << "  exp(lgamma)   " << exp(lgamma(1.+2.)) << std::endl;
	std::cout << "lgamma " << lgamma(2.*gamma+1.) << "  exp(lgamma)   " << exp(lgamma(2.*gamma+1.)) << std::endl;
	
	// output grid and vectors for z
	std::ofstream gridz1out("gz1");
	for (i=0;i<sizeR;i++) {
		gridz1out<<gridz1(i)<<" ";    
		for (j=0;j<sizeR;j++){
			gridz1out<<pow(z1(j,i)*weightR(i),2.)/gridz1(i)/alphaR<<" ";
		}
		gridz1out<<std::endl;
	}
	return;
}

vector tridiagMorse(int sizeR,double De,double mu,double frr,matrix &z,matrix &p2,double alpha, double gamma, int Ne)
{
  int i,j,k;
  double speed_of_light=137.0359895; /* in au */
  double beta,omega,x,Pq_factor,n;
 
  // these come directly from the formulas in the paper                    
  // OMEGA
  /* determination of omega */
  omega=sqrt(frr/mu)/(2.*M_PI*speed_of_light);

  //x
  /* determination of x */
  x=(-alpha*alpha/(4.*M_PI*speed_of_light*mu));

  // BETA
  /* computation of beta */
  beta=sqrt(-2.*x/omega);

  /* construction of the z tridiagonal matrix */

  matrix zeros(sizeR,sizeR);
  z=zeros;
  for (i=0;i<sizeR-1;i++) {
    z(i,i)=2.*(double)i+ 2.*gamma +1.;        // i is n in the paper, so this is <chi_n|z|chi_n>
    z(i,i+1)=(-sqrt((double)(i+1)*((double)(i+1)+2.*gamma)));  // <chi_n|z|chi_(n-1) ... note the shift
    z(i+1,i)=z(i,i+1);
  }
  i=sizeR-1; z(i,i)=2.*(double)i+ 2.*gamma +1.;
  
  vector zdiag=diag(z);
  // z now contains the eigenvectors, i.e, <n|alpha>

  /* conversion of z to r */
  //re=(log(zdiag(sizeR-1)*alpha/(2.*sqrt(2.*mu*De)))/alpha)+4.2;
  
  
 //  for (i=0;i<sizeR;i++)
//     zdiag(i)=re-(log(zdiag(i)*alpha/(2.*sqrt(2.*mu*De)))/alpha);
  
  /* construction of kinetic energy matrix in the fbr*/
  Pq_factor=sqrt(frr/mu)/2.;

  for (i=0;i<sizeR-2;i++) {
    n=(double) i;
    p2(i,i)=Pq_factor*beta*beta/4.*(2.*n*(n+2.*gamma+1.)+2.*gamma+1.);   // why add Pq_factor??
    p2(i,i+2)=Pq_factor*(-beta*beta/4.)*sqrt((n+2.)*(n+1.)*(n+2.*gamma+2.)
					     *(n+2.*gamma+1.));
    p2(i+2,i)=p2(i,i+2);
  }
  i=(sizeR-2);n=(double)i;
  p2(i,i)=Pq_factor*beta*beta/4.*(2.*n*(n+2.*gamma+1.)+2.*gamma+1.);
  i=(sizeR-1);n=(double)i;
  p2(i,i)=Pq_factor*beta*beta/4.*(2.*n*(n+2.*gamma+1.)+2.*gamma+1.);
  /* transformation to the dvr */
  p2=transpose(z)*p2*z;
  // T_n_alpha
  return zdiag;
}

void addtobasis(vector &v,matrix &B,int n,int col)
{
  for(int i=0;i<n;i++)
    B(i,col)=v(i);
  
  return;
}

