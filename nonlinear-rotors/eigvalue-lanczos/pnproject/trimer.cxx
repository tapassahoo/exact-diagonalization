#include "peckeris.h"
#include "random.h"
#include "mbpol.h"
static double betaGlobal;
void lanczosvectors(vector &alpha,vector &beta,vector &beta2,int niter,
					vector &eval,int ngood,matrix &evtr);
void EVanalysis(vector &grid,int size,int nconv,vector &ARv,double Ri,double Rf,
				int basistype,int size3d, diagmat &Rfunc,diagmat &rfunc,
				diagmat &R2func,diagmat &r2func,diagmat &sqrtweight);
void densityanalysis(vector &grid,int size,int nconv,vector &ARv,double Ri,double Rf,
					 int basistype,int size3d, diagmat &Rfunc,diagmat &rfunc,
					 diagmat &R2func,diagmat &r2func,diagmat &sqrtweight,
					 vector &eval);
double silvera(double rval);
double buck(double rval);
double buck1(double rval);
double buckpigs(double rval);
static const double amutoau=1./5.4857989586762187e-4; // amu to au
static const double au_to_amu = 5.4857989586762187e-4; // au to amu
static const double BohrToA=0.529177249; // Bohr to angstrom
static const double HtoW=2.19474631371017e5;//hatree to reciprical centimeter
static const double WavToMHz=29979.2458;
static const double hatokJmol=2625.;
static const double MofH2 = 2.015650642; // nist value
static const double MofD2 = 2.0141017780*2.;// nist
static const double MofHe4 = 4.0026032497; // nist mass of He in amu
main(int argc,char **argv)
{
	int i,j,k,ip,jp,kp,row,n;
	
	// read the input
	if (argc != 11) {
		std::cerr<<"usage: "<<argv[0]<<" <basis> <niter> <V ceiling> <Rinitial> <Rfinal> <a Jacobi> <b Jacobi>";
		std::cerr<<" <basis: HO (0), Jacobi (1) Jacobi t (2) <symmetry: E (0) ; A1 (1); A2 (2); none (>2)> "; 
		std::cerr<<" <H2_2_Silvera:0 D2_2_Silvera:1 H2_2_Buck:2  D2_2_Buck:3 Tom(4)> "<<std::endl;
		exit(0);
	}
	double *pos;
	pos=new double[9];
	size_t nw=1;
	for (i=0;i<3;i++)
		pos[i]=0.;	
	for (i=0;i<2;i++)
		pos[i+3]=0.;	
	pos[3+2]=1.;
	for (i=0;i<3;i++)
		pos[i+6]=0.;	
	pos[6]=1.;	
	std::cout<<"mbpol energy"<< x2o::mbpol()(nw,pos)<<std::endl; 

	exit(0);
	int size=atoi(argv[1]);
	int niter=atoi(argv[2]);
	double Vceil=atof(argv[3]);
	double Ri=atof(argv[4])*atob;
	double Rf=atof(argv[5])*atob;

	double aJacobi=atof(argv[6]);
	double bJacobi=atof(argv[7]);
	int basistype=atoi(argv[8]);
	int symmetrylabel=atoi(argv[9]);
	int system=atoi(argv[10]);
	Vceil/=hatocm;
	int size3d=size*size*size;
	
	for (i=0;i<argc;i++)
		std::cout<<argv[i]<<" ";
	std::cout<<std::endl;
	
	//  int H2=0;
	//double mass=2.01612085117*amutoau;
	double massPIGS=.5/(12.030159*kB*atob*atob);
	double mass;
	if (system == 0 || system==2)
		mass=MofH2/au_to_amu;
	else
		if (system == 1 || system==3)
			mass=MofD2/au_to_amu;
	else {
		if (system==4)
			mass=MofH2/au_to_amu;
		else {
		std::cerr<<"wrong system type!"<<std::endl;
		exit(0);
		}
	}
	//mass=massPIGS;
	
	double mass1=mass;
	double mass2=mass;
	double mass3=mass;
	
	// matrix representation of x and p
	matrix x(size,size);
	matrix ddx=x;
	matrix ddx2=x;
	
	if (basistype == 0) {
		x=xharmonic(size);
		ddx=pharmonic(size);
	}
	if (basistype == 1) {
		x=xJacobi(aJacobi,bJacobi,size);
		ddx=DJacobi(aJacobi,bJacobi,size);
	}
	if (basistype == 2) {
		x=xJacobi(aJacobi,bJacobi,size);
		ddx=BJacobi(aJacobi,bJacobi,size);
	}
	
	ddx2=(-1.)*(transpose(ddx)*ddx);
	
	matrix T=x;
	vector grid=diag(T);  
	
	double xmax=1.01*grid(size-1);
	
	if (basistype == 3) 
		grid=CMgrid(size,Rf-Ri,Ri);
	
	// case of hermite
	
	// fill matrices
	matrix ddr=transpose(T)*ddx*T;
	matrix ddrE(size,size);
	
	double r1,r2,r3;
	double scalefactor;
	if (basistype == 0)
		scalefactor=pow(2.*xmax/(Rf-Ri),2.);
	if (basistype == 1 || basistype == 2)
		scalefactor=pow(2./(Rf-Ri),2.); 
	
	
	// sqrtweight means W_a^(-1/2) ? 
	diagmat sqrtweight(size);
	if (basistype == 0) {
		for (i=0;i<size;i++) {
			sqrtweight(i)=pow(M_PI,-.25)*exp(-grid(i)*grid(i)/2.)/T(0,i);
		}
	}
	if (basistype != 0) {
		for (i=0;i<size;i++) {
			double theta0=pow(1.-grid(i),aJacobi/2.)*pow(1.+grid(i),bJacobi/2.)/sqrt(pow(2.,aJacobi+bJacobi+1.)/(aJacobi+bJacobi+1.)); // only for b=0
			sqrtweight(i)=theta0/T(0,i);
		}
	}
	
	//matrix derivee=((ddr)*transpose(T));
	
	diagmat extraV(size3d);
	diagmat V(size3d);
	int sizeMB=(size*(size-1)*(size-2))/6+size*size;
	std::cout<<sizeMB<<std::endl;
	double VMB;
	diagmat O(size3d);
	diagmat Rfunc(size3d);
	diagmat rfunc(size3d);
	diagmat R2func(size3d);
	diagmat r2func(size3d);

	int i_index;
	int j_index;
	int k_index;

	std::ifstream potin("Pekeris_PES_L6_Nu0.dat");
	for (int index=0;index<sizeMB;index++){
		potin>>i_index>>j_index>>k_index>>VMB;
		i_index-=1;
		j_index-=1;
		k_index-=1;
		VMB/=HtoW;
		V((i_index*size+j_index)*size+k_index)=VMB;
		V((i_index*size+k_index)*size+j_index)=VMB;
		V((k_index*size+j_index)*size+i_index)=VMB;
		V((j_index*size+i_index)*size+k_index)=VMB;
		V((k_index*size+i_index)*size+j_index)=VMB;
		V((j_index*size+k_index)*size+i_index)=VMB;
	}

	for (i=0;i<size;i++) 
		std::cout<<Ri+(grid(i)+1.)*(Rf-Ri)/2.<<" "<<V((i*size+i)*size+i)<<std::endl;
	diagmat G11(size3d);
	diagmat G22(size3d);
	diagmat G33(size3d);
	diagmat G12(size3d);
	diagmat G13(size3d);
	diagmat G23(size3d);
	
	diagmat Va(size3d);
	for (i=0;i<size;i++) {   
		for (j=0;j<size;j++) {      
			for (k=0;k<size;k++) {	
				if (basistype == 1 || basistype ==2) {
					r1=Ri+(grid(i)+1.)*(Rf-Ri)/2.;
					r2=Ri+(grid(j)+1.)*(Rf-Ri)/2.;
					r3=Ri+(grid(k)+1.)*(Rf-Ri)/2.;
				}
				if (basistype == 0) {
					r1=Ri+(grid(i)/xmax+1.)*(Rf-Ri)/2.;
					r2=Ri+(grid(j)/xmax+1.)*(Rf-Ri)/2.;
					r3=Ri+(grid(k)/xmax+1.)*(Rf-Ri)/2.;
				}
				
				double R12=r1+r2;
				double R23=r2+r3;
				double R31=r1+r3; double R13=R31;
				
				// simple (r1+r2+r3)/3 function below
				rfunc((i*size+j)*size+k)=(r1+r2+r3)/3.;
				r2func((i*size+j)*size+k)=(r1*r1+r2*r2+r3*r3)/3.;
				//also (R12+R23+R31)/3
				Rfunc((i*size+j)*size+k)=(R12+R23+R31)/3.;	
				R2func((i*size+j)*size+k)=(R12*R12+R23*R23+R31*R31)/3.;	
				
				//O((i*size+j)*size+k)=(pow(r1,3.)+pow(r2,3.)+pow(r3,3.))/(r1*r2*r3);
				//O((i*size+j)*size+k)=exp(-.1*(r1*r1+r2*r2+r3*r3));
				//O((i*size+j)*size+k)=r1*r2*r3;
				O((i*size+j)*size+k)=(R12+R13+R23)/3.;
				
				if (system == 0 || system==1)
					V((i*size+j)*size+k)=silvera(R12)+silvera(R23)+silvera(R13);
				else
					if (system ==2 || system==3){	
						V((i*size+j)*size+k)=buck(R12)+buck(R23)+buck(R13);
					}
				else {
					if (system == 4)
						i_index++;
					else {
						std::cerr<<"wrong system type"<<std::endl;
						exit(0);
					}
				}
				
				if ( V((i*size+j)*size+k) >= Vceil)  V((i*size+j)*size+k)=Vceil;
				
				G12((i*size+j)*size+k)=scalefactor*(-r1*r2/(mass3*R31*R23));
				G13((i*size+j)*size+k)=scalefactor*(-r1*r3/(mass2*R12*R23));
				G23((i*size+j)*size+k)=scalefactor*(-r2*r3/(mass1*R12*R31));
				G11((i*size+j)*size+k)=scalefactor
					*r1*(
						 (r1+r2+r3)/(mass1*R12*R31)
						 +r2/(mass3*R31*R23)
						 +r3/(mass2*R23*R12));
				G22((i*size+j)*size+k)=scalefactor
					*r2*(
						 (r1+r2+r3)/(mass2*R23*R12)
						 +r3/(mass1*R12*R13)
						 +r1/(mass3*R13*R23));	
				G33((i*size+j)*size+k)=scalefactor
					*r3*(
						 (r1+r2+r3)/(mass3*R13*R23)
						 +r1/(mass2*R23*R12)
						 +r2/(mass1*R12*R13));
				
				
				if (basistype == 2 ) {
					G11((i*size+j)*size+k)/=((1.-grid(i)*grid(i))*(1.-grid(i)*grid(i)));
					G12((i*size+j)*size+k)/=((1.-grid(i)*grid(i))*(1.-grid(j)*grid(j)));
					G13((i*size+j)*size+k)/=((1.-grid(i)*grid(i))*(1.-grid(k)*grid(k)));
					G22((i*size+j)*size+k)/=((1.-grid(j)*grid(j))*(1.-grid(j)*grid(j)));
					G23((i*size+j)*size+k)/=((1.-grid(j)*grid(j))*(1.-grid(k)*grid(k)));
					G33((i*size+j)*size+k)/=((1.-grid(k)*grid(k))*(1.-grid(k)*grid(k)));
					
					
				}
				Va((i*size+j)*size+k)=(pow(R23,4.)/mass1+pow(R31,4.)/mass2+pow(R12,4.)/mass3)
					/(8.*pow(R12*R23*R31,2.));	
				//if ( Va((i*size+j)*size+k) >= Vceil)  Va((i*size+j)*size+k)=Vceil;
			}
		}
	}
	
	
	
	// set the c's to zero
	diagmat VaplusV=Va+V;
	
	double emin=-30./HtoW;
	std::cout<<"emin= "<<emin/kB<<std::endl;
	double emax=0.;
	//double emax=-emin;
	matrix transposeddr=transpose(ddr);
	
	Rand *randomseed = new Rand(1);
	
	// PI lanczos
	int ngood;
	
	vector r(size3d);
	vector evalerr(niter);
	vector eval(niter);
	vector alpha(niter);
	vector beta(niter+1);
	vector beta2(niter+1);
	
	vector v0(size3d);
	//std::ofstream vout("v");
	v0(((size/3)*size+size/2)*size+size/4)=1.;
	// no copies with this kind of starting vector
	for (i=0;i<size;i++)
		for (j=0;j<size;j++)
			for (k=0;k<size;k++){
				v0((i*size+j)*size+k)=2.*(randomseed->frand()-.5);
			}
				for (i=0;i<size;i++) {
					for (j=0;j<size;j++) { 
						if (basistype == 0) {
							r1=Ri+(grid(i)/fabs(grid(0))+1.)*(Rf-Ri)/2.;
							r2=Ri+(grid(j)/fabs(grid(0))+1.)*(Rf-Ri)/2.;
						}
						else {
							r1=Ri+(grid(i)+1.)*(Rf-Ri)/2.;
							r2=Ri+(grid(j)+1.)*(Rf-Ri)/2.;
						}
						
						//   vout<<r1/atob<<" "<<r2/atob<<" "<<v0((i*size+j)*size+size/2)<<" ";
						//  vout<<V((i*size+j)*size+size-1)*hatocm;
						// vout<<" "<<V((i*size+j)*size+size/2)*hatocm;
						//    vout<<" "<<V((i*size+j)*size+size/3)*hatocm;
						//   vout<<" "<<V((i*size+j)*size+size/4)*hatocm;
						//  vout<<" "<<V((i*size+j)*size+size/5)*hatocm;
						//   vout<<" "<<V((i*size+j)*size+size/6)*hatocm;
						//  vout<<" "<<V((i*size+j)*size)*hatocm<<endl;
					}
				}
				
				normalise(v0);
	// analysis of the starting lanczos vector
	// Jason's test of symmetry
	double normofJason=v0*v0;
	if (fabs(normofJason-1.) >= 1.e-6) 
		std::cout<<"problem with symmetry for state "<<i<<std::endl;
	vector u=v0;
	PIA1(u,size);
	normofJason=u*u;
	if (fabs(normofJason-1.) <= 1.e-6)
		std::cout<<"v0 is an A1 vector"<<std::endl;
	u=v0;
	PIE(u,size);
	normofJason=u*u;
	if (fabs(normofJason-1.) <= 1.e-6)
		std::cout<<"v0 is an E "<<std::endl;
	u=v0;
	PIA2(u,size);
	normofJason=u*u;
	if (fabs(normofJason-1.) <= 1.e-6)
		std::cout<<"v0 is an A2 "<<std::endl;
	
	if (symmetrylabel == 0)  PIE(v0,size);
	if (symmetrylabel == 1)  PIA1(v0,size);
	if (symmetrylabel == 2)  PIA2(v0,size);
	std::cout<<"norm of v0 "<<v0*v0<<std::endl;
	// the normalization below is necessary
	normalise(v0);
	
	vector v0keep=v0;
	for (i=0;i<size3d;i++) r(i)=0.;
	//vector v=v0;
    std::cout<<"allocate more memory"<<std::endl;
    matrix Vmat1(size*size,size);
    matrix Vmat2(size*size,size);
    matrix Vmat3(size*size,size);
    matrix W1(size*size,size);
    matrix W2(size*size,size);
    matrix W3(size*size,size);
    std::cout<<"start iterations"<<std::endl;
    for (j=1;j<=niter;j++) {
		
		HpsiJacII(ddr,G11,G22,G33,G12,G13,G23,VaplusV,size,v0,
				  u,Vmat1,Vmat2,Vmat3,W1,W2,W3);
		
		if (symmetrylabel ==0) PIE(u,size);
		if (symmetrylabel ==1) PIA1(u,size);
		if (symmetrylabel ==2) PIA2(u,size);
		
		r=r+u;
		
		alpha(j-1)=v0*r;
		r=r-(alpha(j-1)*v0);
		
		beta2(j)=r*r;
		beta(j)=sqrt(beta2(j));
		r=(1./beta(j))*r; // to get v
		v0=(-beta(j))*v0; // prepare r check minus sign!!!
		
		// purification step (a la Tucker)
		if (symmetrylabel ==0)  PIE(r,size);
		if (symmetrylabel ==1)  PIA1(r,size);
		if (symmetrylabel ==2)  PIA2(r,size);
		
		u=v0;     // swapping
		v0=r;
		r=u;
		if (j%10 == 0)
			std::cout<<"iteration "<<j<<std::endl;
    }                  
    lancbis(niter,eval,evalerr,emin,emax,ngood,alpha,beta,beta2);
    std::cout<<" ngood = "<<ngood<<std::endl;
    std::cout<<"E0 (per partcile)= "<<eval(0)/kB/3.<<std::endl;
    // lanczos report:
    std::ofstream lancout("boundstates.out");
    std::ofstream lanczpeout("states_zpe.out");
    for (i=0;i<ngood;i++) {
		lancout<<eval(i)/kB/3.<<" "<<evalerr(i)/kB<<std::endl;
		lanczpeout<<(eval(i)-eval(0))/kB<<std::endl;
    }
    lancout.flush();
    lancout.close();
    // eigevectors
    
    matrix evtr(niter,ngood);
    lanczosvectors(alpha,beta,beta2,niter,eval,ngood,evtr);
    
    //std::ofstream lvestd::cout("lv");
    vector cumulnorm(ngood);
    for (j=1;j<=niter;j++) {	
		//lvestd::cout<<j<<" ";
		for (n=0;n<ngood;n++) {
			double coeff2=pow(evtr(j-1,n),2.);
			//lvestd::cout<<coeff2<<" ";
			cumulnorm(n)+=coeff2;
			//lvestd::cout<<cumulnorm(n)<<" ";
		}
		//lvestd::cout<<endl;
    }
    
    v0=v0keep;
    vector ARvL(size3d*ngood);
	
    vector testv(size3d);
	
    for (i=0;i<size3d;i++) r(i)=0.;
	
    // lanczos vector coefficent matrix
    for (n=0;n<ngood;n++) cumulnorm(n)=0.;
    for (j=1;j<=niter;j++) {	
		
		// tranform vector	
		for (n=0;n<ngood;n++) {
			double treshold=pow(evtr(j-1,n),2.);
			cumulnorm(n)+=treshold;
			for (row=0;row<size3d;row++){
				double coeff=evtr(j-1,n)*v0(row);	  
				if (cumulnorm(n) <(1.-1.e-16))
					ARvL(row+size3d*n)+=coeff;	
			}
		}
		
		HpsiJacII(ddr,G11,G22,G33,G12,G13,G23,VaplusV,size,v0,
				  u,Vmat1,Vmat2,Vmat3,W1,W2,W3);      
		if (symmetrylabel ==0) PIE(u,size);
		if (symmetrylabel ==1) PIA1(u,size);
		if (symmetrylabel ==2) PIA2(u,size);      
		r=r+u;       
		alpha(j-1)=v0*r;
		r=r-(alpha(j-1)*v0);
		beta2(j)=r*r;
		beta(j)=sqrt(beta2(j));
		r=(1./beta(j))*r; // to get v
		v0=(-beta(j))*v0; // prepare r check minus sign!!!  
		
		
		
		// purification step (a la Tucker)
		if (symmetrylabel ==0)  PIE(r,size);
		if (symmetrylabel ==1)  PIA1(r,size);
		if (symmetrylabel ==2)  PIA2(r,size);      
		u=v0;     // swapping
		v0 =r;
		r=u;	
		
		if (j%100 == 0)
			std::cout<<"iteration "<<j<<std::endl;
    }         
	
    vector psi0(size3d);
    vector V0n(ngood);
    std::cout<<"matrix elements"<<std::endl;
    int tsteps=256;
	//    double deltatau=1./(eval(1)-eval(0));
    double deltatau = (1.4/tsteps)/kB;
    double projectiontime=deltatau*(double)tsteps;
    std::cout<<"projection time= "<<projectiontime*kB<<" 1/K"<<std::endl;
    vector Ctau(tsteps);
    // purification step of all eigenvectors
    for (n=0;n<ngood;n++) {
		for (row=0;row<size3d;row++)
			r(row)=ARvL(row+size3d*n);
		//std::cout<<"PRE norm of psi_"<<n<<" = "<<r*r<<endl;
		if (symmetrylabel ==0)  PIE(r,size);
		if (symmetrylabel ==1)  PIA1(r,size);
		if (symmetrylabel ==2)  PIA2(r,size);  
		//std::cout<<"POST norm of psi_"<<n<<" = "<<r*r<<endl;
		if (n==0) psi0=r;
		
		V0n(n)=psi0*(O*r);
		//V0n(n)/=V0n(0);
		for (int t=0;t<tsteps;t++) {
			double tau=(double)t*deltatau;
			Ctau(t)+=pow(V0n(n),2.)*exp(-tau*(eval(n)-eval(0)));
		}
		//std::cout<<n<<" "<<pow(V0n(n),2.)<<endl;
    }
	
    double O_00=psi0*(O*psi0);
    double O2_00=psi0*(O*O*psi0);
	std::cout.precision(16);	
    std::cout<<" <0|O|0> : "<<O_00/atob<<" A"<<std::endl;
    std::cout<<" <0|O^2|0> : "<<O2_00/atob/atob<<" A^2"<<std::endl;
    std::cout<<" |<0|O|0>|^2 : "<<O_00*O_00/atob/atob<<" A^2"<<std::endl;
	
    std::ofstream Ctauout("Ct");
    std::ofstream V0nout("O_0n");
    for (n=0;n<ngood;n++) V0nout<<n<<" "<<pow(V0n(n)/atob,2.)<<std::endl;
    for (int t=0;t<tsteps;t++) {
	Ctauout.precision(16);
		//Ctauout<<(double)t*deltatau*kB<<" "<<Ctau(t)<<" "<<(Ctau(t)-O_00*O_00)<<" "<< -log(Ctau(t))<<" "<<-log(Ctau(t))+(double)t*deltatau*eval(0) <<endl;
	Ctauout<<(double)t*deltatau<<" "<<(double)t*deltatau*kB<<" "<<Ctau(t)/atob/atob<<" ";
	Ctauout<<exp(-((double)t*deltatau*eval(0)))*(Ctau(t)-(O_00*O_00))<<" ";
	Ctauout<<-log(Ctau(t)/(O_00*O_00)-1.)/((double)t*deltatau)+eval(0)<<" ";
	Ctauout<<-log(fabs(Ctau(t)-O_00*O_00))/((double)t*deltatau)+eval(0);
	Ctauout<<" "<<.5*(-log(Ctau(t)/(O_00*O_00)-1.)/((double)t*deltatau)+eval(0)-log(Ctau(t)-O_00*O_00)/((double)t*deltatau)+eval(0)) <<std::endl;
    }
    EVanalysis(grid,size,ngood,ARvL,Ri,Rf,
			   basistype,size3d,Rfunc,rfunc,
			   R2func,r2func,sqrtweight);    
    std::cout <<"E0 : "<<eval(0)<<std::endl;
    std::cout <<"E1 : "<<eval(1)<<std::endl;
    std::cout <<"E1-E0 : "<<eval(1)-eval(0)<<std::endl;
    std::cout <<"E2-E0 : "<<eval(2)-eval(0)<<std::endl;
    std::cout<<"end of program"<<std::endl;
}
void lanczosvectors(vector &alpha,vector &beta,vector &beta2,int niter,
					vector &eval,int ngood,matrix &evtr)
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
    std::ofstream evout("ev");
    int column1=0;  
    int column2=1;
    int column3=2;
    int column4=3;
    int column5=4;
    int column6=5;
    
    std::ofstream drout("dr");
    std::ofstream Vrout("vr");
    
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
			evout<<std::endl;
		}  
		for ( nc=0;nc<nconv;nc++) Dr(nc) *=pow(sqrtweight(i),2.);
		drout<<r1/atob<<" ";
		for ( nc=0;nc<nconv;nc++) {
			if (basistype == 0)
				drout<<Dr(nc)/((Rf-Ri)/(grid(size-1)-grid(0))/atob)<<" ";
			else
				drout<<Dr(nc)/(.5*(Rf-Ri)/atob)<<" ";
		}
		drout<<std::endl;
    }
    for ( nc=0;nc<nconv;nc++) {
		std::cout<<"<R>_"<<nc<<" = "<<Ravg(nc)/atob<<" A ; ";
		std::cout<<"<r>_"<<nc<<" = "<<ravg(nc)/atob<< " A"<<std::endl;
		std::cout<<"<R^2>_"<<nc<<" = "<<R2avg(nc)/atob/atob<<" A^2 ; ";
		std::cout<<"<r^2>_"<<nc<<" = "<<r2avg(nc)/atob/atob<< " A^2"<<std::endl;
		std::cout<<"(<R^2>-<R>^2)_"<<nc<<" = "<<(R2avg(nc)-Ravg(nc)*Ravg(nc))/atob/atob<<" A^2 ; ";
		std::cout<<"(<r>^2-<r^2>)_"<<nc<<" = "<<(r2avg(nc)-ravg(nc)*ravg(nc))/atob/atob<< " A^2"<<std::endl;
		std::cout<<"(\% equilateral)_"<<nc<<" = "<<countequilateral(nc)*100.<<std::endl;
		std::cout<<"(\% linear)_"<<nc<<" = "<<countlinear(nc)*100.<<std::endl;
		std::cout<<"(\% isoceles)_"<<nc<<" = "<<countisoceles(nc)*100.<<std::endl;
		std::cout<<"(\% scalene)_"<<nc<<" = "<<countscalene(nc)*100.<<std::endl;
		std::cout<<"(\% all)_"<<nc<<" = "<<countall(nc)*100.<<" ";
		std::cout<<(countscalene(nc)+countisoceles(nc)+countlinear(nc)+countequilateral(nc))*100.<<std::endl;
    }
    return;
}

//silvera-goldman potential for molecular hydrogen
double silvera(double r){
	double f,a2i,a6i,rtm,V;
	//  double d=8.321;
	//  double epsilon=3.157865e+05;
	double a=1.713;
	double alpha=1.567;
	alpha=1.5671;
	double beta=-0.00993;
	double c6=12.14;
	double c8=215.2;
	double c9=-143.1;
	double c10=4813.9;
	double rm=3.41;
	rm*=atob;
	double d=1.28*rm; 
	//r=r/rm;
	a2i=1/(r*r);
	a6i=a2i*a2i*a2i;
	rtm=((d/r)-1.0);
	f = (r < d) ? exp(-rtm*rtm) : 1.;
	//  V = epsilon*(exp(a-alpha*r+beta*r*r)-f*a6i*(c6+a2i*(c8+c9/r+a2i*c10)));
	V = (exp(a-alpha*r+beta*r*r)-f*a6i*(c6+a2i*(c8+c9/r+a2i*c10)));
	return V;	
}

//Buck Potential for molecular Hydrogen
double buck(double r){
	double f,V;
	double a=101.4;
	double beta = 2.779;
	double alpha=0.08;
	double c6=7.254;
	double c8=36.008;
	double c10=225.56;
	double d=5.102;
	r/=atob;
	f = 1.0;
	if (r <= d) f = exp(-1*pow((d/r-1),2));
	V = (1./27.2113845)*(a*exp((-1.*beta*r)-(alpha*pow(r,2))) - f*((c6/pow(r,6)) + (c8/pow(r,8)) + (c10/pow(r,10))));
	return V;
}
double buck1(double r){
	double f,V;
	double a=101.4;
	double beta = 2.779;
	double alpha=0.08;
	double c6=7.254;
	double c8=36.008;
	double c10=225.56;
	double d=5.102;
	r/=atob;
	f = 1.0;
	if (r <= d) f = exp(-1*pow((d/r-1),2));
	V = 11604.505*kB*(a*exp((-1.*beta*r)-(alpha*pow(r,2))) - f*((c6/pow(r,6)) + (c8/pow(r,8)) + (c10/pow(r,10))));
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

