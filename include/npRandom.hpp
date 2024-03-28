#ifndef NP_RANDOM_HPP
#define NP_RANDOM_HPP
#include "npArrayCpu.hpp"
#include<time.h>
#include<type_traits>
#include<random>
#include<iostream>

namespace np {
	class Random {
	public:
		//for uniform distribution
		template<typename TP>
		static ArrayCpu<TP> rand(int rows = 1, int cols = 1, int lo = 0, int hi = 1, unsigned long long seed = static_cast<unsigned long long>(time(nullptr)));
		template<typename TP>
		static ArrayCpu<TP> rand(int rows, int cols, unsigned long long seed);

		//for normal distribution

		template<typename TP>
		static ArrayCpu<TP> randn(int rows=1, int cols=1, unsigned long long seed = static_cast<unsigned long long>(time(nullptr)));

	};

	template<typename TP>
	static ArrayCpu<TP> Random::rand(int rows, int cols, int lo, int hi, unsigned long long seed) {
		auto A = ArrayCpu<TP>(rows, cols);
		std::default_random_engine generator(seed);
		std::cout << (std::is_same<int, TP>::value);
		if (std::is_integral<TP>::value) {
			std::uniform_int_distribution<int> distribution(lo, hi);
			for (int i = 0; i < rows * cols; i++) {
				A.mat[i] = distribution(generator);
			}
			return A;
		}
		else {
			std::uniform_real_distribution<double> distribution(lo, hi);
			for (int i = 0; i < rows * cols; i++) {
				A.mat[i] = distribution(generator);
			}
			return A;

		}
	}
	
	template<typename TP>
	static ArrayCpu<TP> Random::rand(int rows, int cols, unsigned long long seed) {
		auto A = ArrayCpu<TP>(rows, cols);
		std::default_random_engine generator(seed);

		if (std::is_integral<TP>::value) {
			std::uniform_int_distribution<int> distribution(0, 1);
			for (int i = 0; i < rows * cols; i++) {
				A.mat[i] = distribution(generator);
			}

		}
		else {
			std::uniform_real_distribution<double> distribution(0, 1);
			for (int i = 0; i < rows * cols; i++) {
				A.mat[i] = distribution(generator);
			}
			return A;

		}
	}

	template<typename TP>
	static ArrayCpu<TP> Random::randn(int rows, int cols, unsigned long long seed) {
		auto A = ArrayCpu<TP>(rows, cols);
		std::default_random_engine generator(seed);
		std::normal_distribution<double>distribution(0, 1);
		for (int i = 0; i < rows * cols; i++) {
			A.mat[i] = distribution(generator);
		}
		return A;
	}

}
#endif