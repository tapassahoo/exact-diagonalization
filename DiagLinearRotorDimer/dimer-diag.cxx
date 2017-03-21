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

double Vinit();
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
void get_sizes(int jmax, int skip, int *sizes);
double basisPjm(int j,int m,double w,double x);
double basisfunctionP(int m,double w,double phi);
double basisfunctionPRe(int m,double w,double phi);
double basisfunctionPIm(int m,double w,double phi);
double PotFunc(double, double, double, double, double);
double RelativeAngle(double, double, double);

int main(int argc,char **argv) 
{
    time_t totalstart,totalend,callpotstart,callpotend,diagstart,diagend;
    time (&totalstart);
    char timemsg[100];
	
    double Rpt          = atof(argv[1]);
    double DipoleMoment = atof(argv[2]);
    int sizej           = atoi(argv[3]);
    int jmax            = sizej;//2*(sizej-1);
	double B            = 20.9561*CMRECIP2KL;

    int size_theta1 = 2.*jmax+5;
	int size_theta2 = 2.*jmax+5;
    int size_phi1   = 2.*(2.*jmax+7); // 
    int size_phi2   = 2.*(2.*jmax+7); // 

    vector weights_theta1(size_theta1);
    vector weights_theta2(size_theta2);
    vector weights_phi1(size_phi1); 
    vector weights_phi2(size_phi2); 

    vector grid_theta1 = thetagrid(size_theta1, weights_theta1);
    vector grid_theta2 = thetagrid(size_theta2, weights_theta2);
    vector grid_phi1   = phigrid(size_phi1, weights_phi1);
    vector grid_phi2   = phigrid(size_phi2, weights_phi2);

    // evaluate basis fxns on grid points
    int sizes[2];
	int skip = 1;
    get_sizes(jmax, skip, sizes);

    ofstream logout("log");

    logout<<"  // test orthonormality of theta and phi basis"<<endl;
    for (int j1 = 0; j1 <= jmax; j1++)
    {
        if (j1%skip) continue; // skip iteration if j1 odd

        for (int m1 = -j1; m1 <= j1; m1++) 
        {
            for (int j2 = 0; j2 <= jmax; j2++)
            {
	            if (j2%skip) continue; // skip iteration if j2 odd

                for (int m2 = -j2; m2 <= j2; m2++) 
                {
	                double sumtRe = 0.;
	                double sumtIm = 0.;

	                for (int i = 0; i < size_theta1; i++)
                    {
	                    double theta1 = grid_theta1(i);

	                	double sumpRe = 0.;
	                	double sumpIm = 0.;
	                	for (int j = 0; j < size_phi1; j++)
                        {
	                        double phi1 = grid_phi1(j);

	                        sumpRe += basisfunctionPRe(m1,weights_phi1(j),phi1)*basisfunctionPRe(m2,weights_phi1(j),phi1);
                            sumpRe += basisfunctionPIm(m1,weights_phi1(j),phi1)*basisfunctionPIm(m2,weights_phi1(j),phi1);

	                        sumpIm += basisfunctionPIm(m1,weights_phi1(j),phi1)*basisfunctionPRe(m2,weights_phi1(j),phi1);
                            sumpIm += -basisfunctionPRe(m1,weights_phi1(j),phi1)*basisfunctionPIm(m2,weights_phi1(j),phi1);
						}
	                    sumtRe += sumpRe*basisPjm(j1,m1,weights_theta1(i),cos(theta1))*basisPjm(j2,m2,weights_theta1(i),cos(theta1));
	                    sumtIm += sumpIm*basisPjm(j1,m1,weights_theta1(i),cos(theta1))*basisPjm(j2,m2,weights_theta1(i),cos(theta1));
	                }
	                logout<<" m1 = "<<m1<<" "<<" m2 = "<<m2<<" "<<j1<<" "<<j2<<"           "<<sumtRe<<" +  "<<sumtIm<<endl;
				}
             }
         }
    }

    for (int m1 = -jmax; m1 <= jmax; m1++) 
    {
        for (int m1p = -jmax; m1p <= jmax; m1p++) 
        {
	       	double sumpRe = 0.;
	       	double sumpIm = 0.;
            for (int k = 0; k < size_phi1; k++)
            {
	            double phi1 = grid_phi1(k);
	            sumpRe += basisfunctionPRe(m1,weights_phi1(k),phi1)*basisfunctionPRe(m1p,weights_phi1(k),phi1);
                sumpRe += basisfunctionPIm(m1,weights_phi1(k),phi1)*basisfunctionPIm(m1p,weights_phi1(k),phi1);

	            sumpIm += basisfunctionPIm(m1,weights_phi1(k),phi1)*basisfunctionPRe(m1p,weights_phi1(k),phi1);
                sumpIm += -basisfunctionPRe(m1,weights_phi1(k),phi1)*basisfunctionPIm(m1p,weights_phi1(k),phi1);
            }
            logout<<" m1,m1p = "<<m1<<" "<<m1p<<" "<<sumpRe<<"  +  "<<sumpIm<<endl;      
        }
    }

//----------------------Full Hamiltonian-----------------------------//

    int sizej1j2m1m2 = sizes[1];
    double theta1, theta2, phi1, phi2;
  
    matrix Angle_V(size_theta1*size_theta2, size_phi1*size_phi2);
    matrix Dphi_V(size_theta1*size_theta2, size_phi1*size_phi2);
    for (int i = 0; i < size_theta1; i++)
    {
        theta1 = grid_theta1(i);

        for (int j = 0; j < size_theta2; j++)
        {
            theta2 = grid_theta2(j);

            for (int k = 0; k < size_phi1; k++)
            {
	            phi1 = grid_phi1(k);

            	for (int l = 0; l < size_phi2; l++)
            	{
	                phi2 = grid_phi2(l);
					double dphi  = phi2 - phi1;

            	    Angle_V(i*size_theta2+j, k*size_phi2+l) = RelativeAngle(theta1, theta2, dphi);
            	    Dphi_V(i*size_theta2+j, k*size_phi2+l) = PotFunc(Rpt, DipoleMoment, theta1, theta2, dphi);
				}
            }
        }
    }

    matrix VRe_basis(sizej1j2m1m2,sizej1j2m1m2);
    matrix VIm_basis(sizej1j2m1m2,sizej1j2m1m2);
    matrix ARe_basis(sizej1j2m1m2,sizej1j2m1m2);
    matrix AIm_basis(sizej1j2m1m2,sizej1j2m1m2);

    int index_j1j2m1m2 = 0;
	// bra parts end here
    for (int j1 = 0; j1 <= jmax; j1++)
    {
        if (j1%skip) continue; // skip iteration if j1 odd

        for (int m1 = -j1; m1 <= j1; m1++) 
        {
          	for (int j2 = 0; j2 <= jmax; j2++)
           	{
	           	if (j2%skip) continue; // skip iteration if j2 odd

    		    for (int m2 = -j2; m2 <= j2; m2++) 
    		    {
					
					// ket parts end here
	           		int index_j1j2m1m2p = 0;
	                for (int j1p = 0; j1p <= jmax; j1p++)
                    {
	                    if (j1p%skip) continue; // skip iteration if j1 odd
	            	    for (int m1p = -j1p; m1p <= j1p; m1p++) 
                	    {
	                        for (int j2p = 0; j2p <= jmax; j2p++)
                            {
	                            if (j2p%skip) continue; // skip iteration if j2 odd

	            	            for (int m2p =- j2p; m2p <= j2p; m2p++) 
                	            {
	                                // theta1 quadrature
                                    double sumt1Re = 0.0;
                                    double sumt1Im = 0.0;
                                    double sumt1ReA = 0.0;
                                    double sumt1ImA = 0.0;
	                                for (int i = 0; i < size_theta1; i++)
                                    {
		                                theta1 = grid_theta1(i);

	                                	// theta2 quadrature
                                        double sumt2Re = 0.0;
                                        double sumt2Im = 0.0;
                                        double sumt2ReA = 0.0;
                                        double sumt2ImA = 0.0;
		                                for (int j = 0; j < size_theta2; j++)
                                        {
		                                    theta2 = grid_theta2(j);

                                            double sump1Re = 0.0;
                                            double sump1Im = 0.0;
                                            double sump1ReA = 0.0;
                                            double sump1ImA = 0.0;
		                                    // phi1 quadrature
		                                    for (int k = 0; k < size_phi1; k++)
                                            {
		                                        phi1 = grid_phi1(k);

												double sump2Re = 0.0; 
												double sump2Im = 0.0; 
												double sump2ReA = 0.0; 
												double sump2ImA = 0.0; 
		                                        // phi2 quadrature
		                           	            for (int l = 0; l < size_phi2; l++)
                                   	            {
		                                	        phi2 = grid_phi2(l);
                                                    double dphi   = phi2 - phi1;
                                                    double pot    = Dphi_V(i*size_theta2+j, k*size_phi2+l);
	                                                sump2Re += pot*basisfunctionPRe(m2,weights_phi2(l),phi2)*basisfunctionPRe(m2p,weights_phi2(l),phi2);
                                                    sump2Re += pot*basisfunctionPIm(m2,weights_phi2(l),phi2)*basisfunctionPIm(m2p,weights_phi2(l),phi2);
	                                                sump2Im += pot*basisfunctionPIm(m2,weights_phi2(l),phi2)*basisfunctionPRe(m2p,weights_phi2(l),phi2);
                                                    sump2Im += -pot*basisfunctionPRe(m2,weights_phi2(l),phi2)*basisfunctionPIm(m2p,weights_phi2(l),phi2);

                                                    double Angle  = Angle_V(i*size_theta2+j, k*size_phi2+l);
	                                                sump2ReA += Angle*basisfunctionPRe(m2,weights_phi2(l),phi2)*basisfunctionPRe(m2p,weights_phi2(l),phi2);
                                                    sump2ReA += Angle*basisfunctionPIm(m2,weights_phi2(l),phi2)*basisfunctionPIm(m2p,weights_phi2(l),phi2);
	                                                sump2ImA += Angle*basisfunctionPIm(m2,weights_phi2(l),phi2)*basisfunctionPRe(m2p,weights_phi2(l),phi2);
                                                    sump2ImA += -Angle*basisfunctionPRe(m2,weights_phi2(l),phi2)*basisfunctionPIm(m2p,weights_phi2(l),phi2);
		                            	        }
								                double p1re  = basisfunctionPRe(m1,weights_phi1(k),phi1)*basisfunctionPRe(m1p,weights_phi1(k),phi1);
                                                p1re        += basisfunctionPIm(m1,weights_phi1(k),phi1)*basisfunctionPIm(m1p,weights_phi1(k),phi1);
	                                            double p1im  = basisfunctionPIm(m1,weights_phi1(k),phi1)*basisfunctionPRe(m1p,weights_phi1(k),phi1);
                                                p1im        += -basisfunctionPRe(m1,weights_phi1(k),phi1)*basisfunctionPIm(m1p,weights_phi1(k),phi1);

                                                sump1Re     += sump2Re*p1re - sump2Im*p1im;
                                                sump1Im     += sump2Re*p1im + sump2Im*p1re;

                                                sump1ReA     += sump2ReA*p1re - sump2ImA*p1im;
                                                sump1ImA     += sump2ReA*p1im + sump2ImA*p1re;
		                                    }
		                                    sumt2Re         += sump1Re*basisPjm(j2,m2,weights_theta2(j),cos(theta2))*basisPjm(j2p,m2p,weights_theta2(j),cos(theta2));
		                                    sumt2Im         += sump1Im*basisPjm(j2,m2,weights_theta2(j),cos(theta2))*basisPjm(j2p,m2p,weights_theta2(j),cos(theta2));

		                                    sumt2ReA         += sump1ReA*basisPjm(j2,m2,weights_theta2(j),cos(theta2))*basisPjm(j2p,m2p,weights_theta2(j),cos(theta2));
		                                    sumt2ImA         += sump1ImA*basisPjm(j2,m2,weights_theta2(j),cos(theta2))*basisPjm(j2p,m2p,weights_theta2(j),cos(theta2));
		                                }
		                                sumt1Re             += sumt2Re*basisPjm(j1,m1,weights_theta1(i),cos(theta1))*basisPjm(j1p,m1p,weights_theta1(i),cos(theta1));
		                                sumt1Im             += sumt2Im*basisPjm(j1,m1,weights_theta1(i),cos(theta1))*basisPjm(j1p,m1p,weights_theta1(i),cos(theta1));

		                                sumt1ReA             += sumt2ReA*basisPjm(j1,m1,weights_theta1(i),cos(theta1))*basisPjm(j1p,m1p,weights_theta1(i),cos(theta1));
		                                sumt1ImA             += sumt2ImA*basisPjm(j1,m1,weights_theta1(i),cos(theta1))*basisPjm(j1p,m1p,weights_theta1(i),cos(theta1));
									}
		                            VRe_basis(index_j1j2m1m2,index_j1j2m1m2p)   = sumt1Re;
		                            VIm_basis(index_j1j2m1m2,index_j1j2m1m2p)   = sumt1Im;
		                            ARe_basis(index_j1j2m1m2,index_j1j2m1m2p)   = sumt1ReA;
		                            AIm_basis(index_j1j2m1m2,index_j1j2m1m2p)   = sumt1ImA;
	                                index_j1j2m1m2p++;
								}	
	                        }
	                    }
	                }
	                index_j1j2m1m2++;
	            }
            }
        }
    }
    
    matrix HRe(sizej1j2m1m2,sizej1j2m1m2);
    matrix HIm(sizej1j2m1m2,sizej1j2m1m2);
    matrix VRe(sizej1j2m1m2,sizej1j2m1m2);
    matrix VIm(sizej1j2m1m2,sizej1j2m1m2);
    matrix Hrot(sizej1j2m1m2,sizej1j2m1m2);

    ofstream Vout("V");
    ofstream Hout("H");

    int index_jm12 = 0;
    for (int j1 = 0; j1 <= jmax; j1++) 
    {
        if (j1%skip) continue; 
        for (int m1 = -j1; m1 <= j1; m1++)
        {
            for (int j2 = 0; j2 <= jmax; j2++)
            {
	            if (j2%skip) continue; 
    			for (int m2 = -j2; m2 <= j2; m2++)
    			{
	                int index_jm12p = 0;
	                for (int j1p = 0; j1p <= jmax; j1p++)
                    {
	                    if (j1p%skip) continue; 
	                    for (int m1p = -j1p; m1p <= j1p; m1p++) 
                        {
	                        for (int j2p = 0; j2p <= jmax; j2p++)
                            {
	                            if (j2p%skip) continue; 
	                            for (int m2p = -j2p; m2p <= j2p; m2p++) 
                                {
	      
	                                VRe(index_jm12,index_jm12p) = VRe_basis(index_jm12,index_jm12p);
	                                VIm(index_jm12,index_jm12p) = VIm_basis(index_jm12,index_jm12p);

	                                if (j1 == j1p && j2 == j2p && m1 == m1p && m2 == m2p)
									{
	                				    HRe(index_jm12,index_jm12p) = B*((double)(j1*(j1+1))+(double)(j2*(j2+1)));
	                				    HIm(index_jm12,index_jm12p) = 0.0;
									}
									else
									{
	                				    HRe(index_jm12,index_jm12p) = 0.0;
	                				    HIm(index_jm12,index_jm12p) = 0.0;
									}
	                				index_jm12p++;		       
	                            }
	            			}
						}
      				}
	                index_jm12++;
    			}
  			}
		}
	}

	cout<<index_jm12<<"  index  "<<sizej1j2m1m2<<endl;
    // Diagonalize

    time (&diagstart);
  
    Hrot = HRe;
    HRe = HRe+VRe;

    //  ofstream Hout("H");
	/*
    for (int i = 0; i < sizej1j2m1m2; i++) 
    {
        for (int j = 0; j < sizej1j2m1m2; j++)
        {
            if(HRe(i,j) != HRe(j,i)) cout<< i<< "   "<<j<<"   "<<HRe(i,j)<<" "<<HRe(j,i)<<endl;
            //Hout<< i<< "   "<<j<<"   "<<H(i,j)<<" "<<H(j,i)<<endl;
		}
        //Hout<<endl;
  	}
	*/

	vector ev=diag(HRe);
	time (&diagend);
	double dif2 = difftime (diagend,diagstart);
	sprintf (timemsg, "Elapsed time for diagonalization is %.2lf seconds.", dif2);
	logout<<timemsg<<endl;

	ofstream evalues("ev");
	evalues.precision(3);
	for (int i = 0; i < sizej1j2m1m2; i++)
    {
    	evalues<<ev(i)<<"   ";
    	for (int j = 0; j < sizej1j2m1m2; j++)
        {
      		evalues<<HRe(j,i)<<" ";
		}
     	evalues<<endl;
	}

	matrix avgV    = transpose(HRe)*VRe*HRe;
	matrix avgHrot = transpose(HRe)*Hrot*HRe;
    matrix avgAng  = transpose(HRe)*ARe_basis*HRe;

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
    eigenvalues<< setw(20)<<setfill(' ')<<avgHrot(0,0);
    eigenvalues<< setw(20)<<setfill(' ')<<avgAng(0,0)<<endl;


    time (&totalend);
    double dif3 = difftime (totalend,totalstart);
    sprintf (timemsg, "Total elapsed time is %.2lf seconds.\n", dif3);
    logout<<timemsg<<endl;
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
    if (m>0) value=sqrt(1./(M_PI))*sqrt(w)*cos((double)m*phi);
    if (m==0) value=sqrt(1./(2.*M_PI))*sqrt(w);
    if (m<0) value=sqrt(1./(M_PI))*sqrt(w)*sin((double)abs(m)*phi);
  	return value;
}

double basisfunctionPRe(int m,double w,double phi)
{
	double value;
    value=sqrt(1./(2.0*M_PI))*sqrt(w)*cos((double)m*phi);
  	return value;
}

double basisfunctionPIm(int m,double w,double phi)
{
	double value;
    value=sqrt(1./(2.0*M_PI))*sqrt(w)*sin((double)m*phi);
  	return value;
}

void get_sizes(int jmax, int skip, int *sizes)
{
    int j1,j2,m1,m2;
    int indexmj    = 0;
    int indexmj1j2 = 0;
    for (int j1 = 0; j1 <= jmax; j1++) 
    {
        if (j1%skip) continue; // skip iteration if j odd
        for (int m1 = -j1; m1 <= j1; m1++) 
        {
          	for (int j2 = 0; j2 <= jmax; j2++) 
           	{
	           	if (j2%skip) continue;

    	    	for (int m2 = -j2; m2 <= j2; m2++) 
    		    {
	            	indexmj1j2++;
				}
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
    double Angle          = sin(theta1)*sin(theta2)*cos(phi) - cos(theta1)*cos(theta2);
    return Angle;
}
