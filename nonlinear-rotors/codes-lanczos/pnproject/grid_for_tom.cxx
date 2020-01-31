#include "peckeris.h"
#include "random.h"
main(int argc,char **argv)
{
	int i,j,k,ip,jp,kp,row,n;
	


	if (argc != 6) {
		std::cerr<<"wrong number of arguments; usage: "<<argv[0]<<" size Ri Rf aJacobi bJacobi"<<std::endl;
		exit(0);
	}

	int size=atoi(argv[1]);
        double Ri=atof(argv[2]);
        double Rf=atof(argv[3]);
        double aJacobi=atof(argv[4]);
        double bJacobi=atof(argv[5]);

	matrix	x=xJacobi(aJacobi,bJacobi,size);
	vector grid=diag(x);
	for (i=0;i<size;i++) {
		std::cout<<Ri+(1.+grid(i))/2.*(Rf-Ri)<<std::endl;
	}
}
