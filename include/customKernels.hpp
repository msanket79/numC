#ifndef CUSTOM_KERNELS_HPP
#define CUSTOM_KERNELS_HPP
#include<omp.h>

//adding matrices with broadcasting
template<typename TP>
inline void kernelMatAddMat(const TP* A,const TP* B,TP* C,const int rows,const int cols);

template<typename TP>
inline void kernelMatAddScaler(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatAddVecAlongRows(const TP* A, const TP* B,  TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatAddVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//subtracting matrices with broadcasting
template<typename TP>
inline void kernelMatSubMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatSubScaler(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatSubVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatSubVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//multiplying
template<typename TP>
inline void kernelMatMulMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatMulScaler(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatMulVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatMulVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//dividing
template<typename TP>
inline void kernelMatDivMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatDivScaler(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatDivVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatDivVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//GreaterThan
template<typename TP>
inline void kernelMatGreaterThanMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatGreaterThanScaler(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatGreaterThanVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatGreaterThanVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//GreaterThanEqual
template<typename TP>
inline void kernelMatGreaterThanEqualMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatGreaterThanEqualScaler(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatGreaterThanEqualVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatGreaterThanEqualVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//LessThan
template<typename TP>
inline void kernelMatLessThanMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatLessThanScaler(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatLessThanVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatLessThanVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//LessThanEqual
template<typename TP>
inline void kernelMatLessThanEqualMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatLessThanEqualScaler(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatLessThanEqualVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatLessThanEqualVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//misc
template<typename TP>
void FloatToInt(int* B, TP* mat, int rows, int cols);

template<typename TP>
void IntToFloat(float* B, TP* mat, int rows, int cols);

template<typename TP>
void InitMat(TP* mat, int rows, int cols, TP val);

//function definitions
template<typename TP>
inline void kernelMatAddScaler(const TP* A, const TP b,  TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] + b;
		}
	}
}

template<typename TP>
inline void kernelMatAddMat(const TP* A,const TP* B,TP* C,const int rows,const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
		}
	}
}

template<typename TP>
inline void kernelMatAddVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i*cols+j]=A[i * cols + j] + B[j];
		}
	}
}

template<typename TP>
inline void kernelMatAddVecAlongCols(const TP* A, const TP* B,  TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		int a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] + a;
		}
	}
}

///-------------------------------------------------------
template<typename TP>
inline void kernelMatSubScaler(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] - b;
		}
	}
}

template<typename TP>
inline void kernelMatSubMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] - B[i * cols + j];
		}
	}
}

template<typename TP>
inline void kernelMatSubVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] - B[j];
		}
	}
}

template<typename TP>
inline void kernelMatSubVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		int a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] - a;
		}
	}
}

//--------------------------------------------------------
template<typename TP>
inline void kernelMatMulScaler(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] * b;
		}
	}
}

template<typename TP>
inline void kernelMatMulMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] * B[i * cols + j];
		}
	}
}

template<typename TP>
inline void kernelMatMulVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] * B[j];
		}
	}
}

template<typename TP>
inline void kernelMatMulVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		int a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] * a;
		}
	}
}

//--------------------------------------------------------

template<typename TP>
inline void kernelMatDivScaler(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] / b;
		}
	}
}

template<typename TP>
inline void kernelMatDivMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] / B[i * cols + j];
		}
	}
}

template<typename TP>
inline void kernelMatDivVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] / B[j];
		}
	}
}

template<typename TP>
inline void kernelMatDivVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		int a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] / a;
		}
	}
}

//-------------------------------------------------------

template<typename TP>
inline void kernelMatGreaterThanScaler(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] > b ? 1:0);
		}
	}
}

template<typename TP>
inline void kernelMatGreaterThanMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] > B[i * cols + j] ?1:0);
		}
	}
}

template<typename TP>
inline void kernelMatGreaterThanVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] > B[j]?1:0);
		}
	}
}

template<typename TP>
inline void kernelMatGreaterThanVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		int a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] > a ?1:0);
		}
	}
}

//-------------------------------------------------------

template<typename TP>
inline void kernelMatGreaterThanEqualScaler(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] >= b ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatGreaterThanEqualMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] >= B[i * cols + j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatGreaterThanEqualVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] >= B[j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatGreaterThanEqualVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		int a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] >= a ? 1 : 0);
		}
	}
}

//--------------------------------------------------

template<typename TP>
inline void kernelMatLessThanScaler(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] < b ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatLessThanMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] < B[i * cols + j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatLessThanVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] < B[j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatLessThanVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		int a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] < a ? 1 : 0);
		}
	}
}


//-----------------------------------------------------------------------------

template<typename TP>
inline void kernelMatLessThanEqualScaler(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] <= b ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatLessThanEqualMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] <= B[i * cols + j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatLessThanEqualVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] <= B[j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatLessThanEqualVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		int a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] <= a ? 1 : 0);
		}
	}
}


template<typename TP>
void InitMat(TP* mat, int rows, int cols, TP val) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			mat[i * cols + j] = val;
		}
	}
}
template<typename TP>
void IntToFloat(float* B, TP* mat, int rows, int cols) {

#pragma omp parallel for
	for (int i = 0; i < rows * cols; i++) {
		B[i] = static_cast<float>(mat[i]);
	}

}
template<typename TP>
void FloatToInt(int* B, TP* mat, int rows, int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows * cols; i++) {
		B[i] = static_cast<int>(mat[i]);
	}

}

#endif // !CUSTOM_KERNELS_HPP

