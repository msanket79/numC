#ifndef NP_FUNCTIONS_HPP
#define NP_FUNCTIONS_HPP

#include "npArrayCpu.hpp"

namespace np {
	template<typename TP>
	ArrayCpu<TP> arrange(const TP start,const TP stop,const TP step);

	template <typename TP>
	ArrayCpu<TP> ones(int rows = 1, int cols = 1);

	template<typename TP>
	ArrayCpu<TP> zeros(int rows = 1, int cols = 1);

	template<typename TP>
	ArrayCpu<TP> maximum(ArrayCpu<TP>& A, ArrayCpu<TP>& B);

	template<typename TP>
	ArrayCpu<TP> exp( ArrayCpu<TP>&A);

	template<typename TP>
	ArrayCpu<TP> log( ArrayCpu<TP>&A);

	template<typename TP>
	ArrayCpu<TP> sqrt(ArrayCpu<TP>& A);

	template<typename TP>
	ArrayCpu<TP> square(ArrayCpu<TP>& A);

	template<typename TP,typename TP1>
	ArrayCpu<TP> pow(ArrayCpu<TP>& A, TP1 n);

//function declaration-------------

	template<typename TP>
	np::ArrayCpu<TP> maximum(np::ArrayCpu<TP>& B) {
		//when A is scalaer

		if (this->rows == 1 and this->cols == 1) {

			auto C = ArrayCpu<TP>(B.rows, B.cols);
			kernelMatMaximumScaler<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
			return C;

		}
		else if (B.rows == 1 and B.cols == 1) {

			auto C = ArrayCpu<TP>(this->rows, this->cols);
			kernelMatMaximumScaler<TP>(this->mat, B(0, 0), C.mat, this->rows, this->cols);
			return C;
		}
		else if (this->rows == B.rows && this->cols == B.cols) {

			auto C = ArrayCpu<TP>(this->rows, this->cols);
			kernelMatMaximumMat<TP>(this->mat, B.mat, C.mat, rows, cols);
			return C;
		}
		else if (this->rows == 1 || this->cols == 1) {

			int vecDim = this->rows > this->cols ? this->rows : this->cols;
			//when the second matrix is square matrix then exception
			if (vecDim == B.rows && vecDim == B.cols) {

			}
			else if (vecDim == B.rows) {
				auto C = ArrayCpu<TP>(B.rows, B.cols);
				kernelMatMaximumVecAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
				return C;
			}
			else if (vecDim == B.cols) {
				auto C = ArrayCpu<TP>(B.rows, B.cols);
				kernelMatMaximumVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
				return C;
			}
		}
		else if (B.rows == 1 || B.cols == 1) {

			int vecDim = B.rows > B.cols ? B.rows : B.cols;
			std::cout << vecDim;
			if (vecDim == this->rows && vecDim == this->cols) {

			}
			else if (vecDim == this->rows) {

				auto C = ArrayCpu<TP>(this->rows, this->cols);
				kernelMatMaximumVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
				return C;
			}
			else if (vecDim == this->cols) {
				auto C = ArrayCpu<TP>(this->rows, this->cols);
				kernelMatMaximumVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
				return C;
			}
		}

	}


	template<typename TP>
	np::ArrayCpu<TP> maximum(TP b) {
		auto C = np::ArrayCpu<TP>(this->rows, this->cols);
		kernelMatMaximumScaler<TP>(this->mat, b, C.mat, this->rows, this->cols);
		return C;

	}



	template<typename TP>
	ArrayCpu<TP> arrange(const TP start,const TP stop,const TP step) {
		int cols = ceil((stop - start) / step);
		ArrayCpu<TP> A = ArrayCpu<TP>(1, cols);
		int j = 0;
		for (TP i = start; i < stop; i += step) {
			A(0, j) = i;
			j++;
		}
		return A;


	}
	template <typename TP>
	 ArrayCpu<TP> ones(int rows,int cols) {
		 auto A = ArrayCpu<TP>(rows, cols);
#pragma omp parallel for
		 for (int i = 0; i < rows; i++) {
			 for (int j = 0; j < cols; j++) {
				 A.mat[i * cols + j] = 1;
			 }
		 }
		return A;
	}
	template<typename TP>
	ArrayCpu<TP> zeros(int rows,int cols) {
		auto A = ArrayCpu<TP>(rows, cols,0);
		#pragma omp parallel for
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				A.mat[i * cols + j] = 0;
			}
		}
		return A;
	}
	template<typename TP>
	ArrayCpu<TP> maximum(ArrayCpu<TP>& A, ArrayCpu<TP>& B) {
		auto C = ArrayCpu<TP>(A.rows, A.cols);
		#pragma omp parallel for
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < A.cols; j++) {
				TP a = A(i, j);
				TP b = B(i, j);
				if (a > b) C(i, j) = a;
				else C(i, j) = b;
			}
		}
		return C;
	}
	template<typename TP>
	ArrayCpu<TP> exp(ArrayCpu<TP>&A) {
		auto B = ArrayCpu<TP>(A.rows, A.cols, 0);
		
		#pragma omp parallel for
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < A.cols; j++) {
				B(i, j) = std::exp(A(i, j));
			}
		}
		return B;

	}
	template<typename TP>
	ArrayCpu<TP> log(ArrayCpu<TP>&A) {
		auto B = ArrayCpu<TP>(A.rows, A.cols, 0);
		#pragma omp parallel for
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < A.cols; j++) {
				B(i, j) = std::log(A(i, j));
			}
		}
		return B;
	}

	template<typename TP>
	ArrayCpu<TP> sqrt(ArrayCpu<TP>& A) {
		auto B = ArrayCpu<TP>(A.rows, A.cols, 0);

#pragma omp parallel for
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < A.cols; j++) {
				B(i, j) = std::sqrt(A(i, j));
			}
		}
		return B;

	}

	template<typename TP>
	ArrayCpu<TP> square(ArrayCpu<TP>& A) {
		auto B = ArrayCpu<TP>(A.rows, A.cols, 0);

#pragma omp parallel for
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < A.cols; j++) {
				B(i, j) = std::pow(A(i, j),2);
			}
		}
		return B;

	}

	template<typename TP,typename TP1>
	ArrayCpu<TP> pow(ArrayCpu<TP>& A,TP1 n) {
		auto B = ArrayCpu<TP>(A.rows, A.cols, 0);

#pragma omp parallel for
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < A.cols; j++) {
				B(i, j) = std::pow(A(i, j),n);
			}
		}
		return B;

	}
}
#endif