#include<iostream>
#include<sstream>
#include<math.h> // or <cmath>

int main(int argc, char **argv)
{
	double input=10.0;
    std::cout << "gamma( " << input << " ) = " << tgamma(input) << std::endl;
    return 0;
}
