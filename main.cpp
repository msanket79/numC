#include "include/npArrayCpu.hpp"
#include "include/npRandom.hpp"
#include "include/npFunctions.hpp"
#include<iostream>

int main() {
	// auto A = np::arrange<double>(1,4,2);
	// auto B = np::ArrayCpu<int>(4, 4, 2);

	auto A = np::Random::rand<float>(4096, 4096);
	auto B = np::Random::randn<float>(4096, 4096);

	auto C = A.dot(B);
	// std::cout << B;
	// std::cout << A;
	//auto B = np::ArrayCpu<double>(1, 5,4);
	//std::cout << B;
	//auto C = A < B;
	// auto C =np::pow(A, 2);
	// auto D = np::square(A);
	// std::cout << C;
	// std::cout << D;
	return 0;
	
}



//==
//maximum  with broadcasting
// minimum with broadcasting
//
