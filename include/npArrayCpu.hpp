#ifndef NP_ARRAY_CPU_HPP
#define NP_ARRAY_CPU_HPP
#include<cstring>
#include<cmath>
#include<type_traits>
#include<omp.h>
#include<cblas.h>
#include<iostream>
#include "customKernels.hpp"


namespace np {

template<typename TP>
class ArrayCpu {
public:
    TP* mat;
    int rows, cols;

    ArrayCpu(int rows = 1, int cols = 1,TP val=0);

    ArrayCpu(const ArrayCpu& other) : rows(other.rows), cols(other.cols) {
        mat = new TP[rows * cols];
        std::memcpy(mat, other.mat, rows * cols * sizeof(TP));
    }

    ~ArrayCpu();

    void operator=(const ArrayCpu& other);

    friend std::ostream& operator<<(std::ostream& out, const ArrayCpu<TP>& Arr);

    TP& operator()(int i, int j);

    void operator<<(TP* A);

    ArrayCpu<TP> operator+(ArrayCpu<TP>& B);

    ArrayCpu<TP> operator+(TP b);

    ArrayCpu<TP> operator-(ArrayCpu<TP>& B);

    ArrayCpu<TP> operator-() const;

    ArrayCpu<TP> operator-(TP b);

    ArrayCpu<TP> operator*(ArrayCpu<TP>& B);

    ArrayCpu<TP> operator*(TP b);

    ArrayCpu<TP> operator/(ArrayCpu<TP>& B);

    ArrayCpu<TP> operator/(TP b);

    
    ArrayCpu<TP> operator>(ArrayCpu& B);

    ArrayCpu<TP> operator>(TP b);

    ArrayCpu<TP> operator>=(ArrayCpu& B);

    ArrayCpu<TP> operator>=(TP b);

    ArrayCpu<TP> operator<(ArrayCpu& B);

    ArrayCpu<TP> operator<(TP b);

    ArrayCpu<TP> operator<=(ArrayCpu& B);

    ArrayCpu<TP> operator<=(TP b);

    int at(int i, int j);

    void set(int i, int j, int val);

    ArrayCpu<TP> T();

    ArrayCpu<TP> sum(int axis = -1);

    ArrayCpu<TP> min(int axis = -1);

    ArrayCpu<TP> max(int axis = -1);

    ArrayCpu<TP> argmin(int axis = -1);

    ArrayCpu<TP> argmax(int axis = -1);

    ArrayCpu<TP> dot(ArrayCpu<TP>& B);

    ArrayCpu<TP> Tdot(ArrayCpu<TP>& B);

    ArrayCpu<TP> dotT(ArrayCpu<TP>& B);

    void reshape(int rows, int cols);

};
template<typename TP>
void np::ArrayCpu<TP>::operator=(const ArrayCpu& other) {
    if (this != other) {
        delete[] mat;
        rows = other.rows;
        cols = other.cols;
        mat = new TP[rows * cols];
        std::memcpy(mat, other.mat, rows * cols * sizeof(TP));
    }
}

template<typename TP>
ArrayCpu<TP> ArrayCpu<TP>::operator-() const {
    auto A = ArrayCpu<TP>(this->rows, this->cols);
#pragma omp prallel
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            A(i, j) = this->mat[i * cols + j];
        }
    }
    return A;
}

template<typename TP>
np::ArrayCpu<TP>::ArrayCpu(int rows, int cols, TP val) {
    this->rows = rows;
    this->cols = cols;
    this->mat = (TP*)malloc(sizeof(TP) * rows * cols);
    //memset works fine for only 0 and -1 ,sometimes it works for other values sometime it doesn't 
    //so making my own initialization function
    InitMat<TP>(mat, rows, cols, val);
}

template<typename TP>
np::ArrayCpu<TP>::~ArrayCpu() {
    delete[] mat;
    mat = nullptr;
}

template<typename TP>
void np::ArrayCpu<TP>::operator<<(TP* A) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] = A[i * cols + j];
        }
    }
}

template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::dot(np::ArrayCpu<TP>& B) {
    if (std::is_integral<TP>::value) {
        float* A = new float[rows * cols];
        float* Bmat = new float[B.rows * B.cols];
        float* C = new float[rows * B.cols];
        IntToFloat(A, mat, rows, cols);
        IntToFloat(Bmat, B.mat, B.rows, B.cols);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, B.cols, cols, 1.0, A, cols, Bmat, B.cols, 0.0, C, B.cols);
        auto Ans = np::ArrayCpu<TP>(rows, B.cols);
        FloatToInt(reinterpret_cast<int*>(Ans.mat), C, rows, B.cols);
        return Ans;


    }
    else if (std::is_same<TP, float>::value) {
        auto C = np::ArrayCpu<TP>(rows, B.cols, 0);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, B.cols, cols, 1.0, (float*) (mat), cols, reinterpret_cast<float*>(B.mat), B.cols, 0.0, reinterpret_cast<float*>(C.mat), B.cols);
        return C;
    }
    else if (std::is_same<TP, double>::value) {
        auto C = np::ArrayCpu<TP>(rows, B.cols, 0);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, B.cols, cols, 1.0, reinterpret_cast<double*> (mat), cols, reinterpret_cast<double*>(B.mat), B.cols, 0.0, reinterpret_cast<double*>(C.mat), B.cols);
        return C;

    }
    return NULL;
}

template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::dotT(np::ArrayCpu<TP>& B) {
    if (std::is_integral<TP>::value) {
        float* A = new float[rows * cols];
        float* Bmat = new float[B.rows * B.cols];
        float* C = new float[rows * B.cols];
        IntToFloat(A, mat, rows, cols);
        IntToFloat(Bmat, B.mat, B.rows, B.cols);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, B.cols, cols, 1.0, A, cols, Bmat, B.cols, 0.0, C, B.cols);
        auto Ans = np::ArrayCpu<TP>(rows, B.cols);
        FloatToInt(reinterpret_cast<int*>(Ans.mat), C, rows, B.cols);
        return Ans;


    }
    else if (std::is_same<TP, float>::value) {
        auto C = np::ArrayCpu<TP>(rows, B.cols, 0);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, B.cols, cols, 1.0, reinterpret_cast<float*> (mat), cols, reinterpret_cast<float*>(B.mat), B.cols, 0.0, reinterpret_cast<float*>(C.mat), B.cols);
        return C;
    }
    else if (std::is_same<TP, double>::value) {
        auto C = np::ArrayCpu<TP>(rows, B.cols, 0);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, B.cols, cols, 1.0, reinterpret_cast<double*> (mat), cols, reinterpret_cast<double*>(B.mat), B.cols, 0.0, reinterpret_cast<double*>(C.mat), B.cols);
        return C;

    }
    return NULL;
}

template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::Tdot(np::ArrayCpu<TP>& B) {
    if (std::is_integral<TP>::value) {
        float* A = new float[rows * cols];
        float* Bmat = new float[B.rows * B.cols];
        float* C = new float[rows * B.cols];
        IntToFloat(A, mat, rows, cols);
        IntToFloat(Bmat, B.mat, B.rows, B.cols);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, rows, B.cols, cols, 1.0, A, cols, Bmat, B.cols, 0.0, C, B.cols);
        auto Ans = np::ArrayCpu<TP>(rows, B.cols);
        FloatToInt(reinterpret_cast<int*>(Ans.mat), C, rows, B.cols);
        return Ans;


    }
    else if (std::is_same<TP, float>::value) {
        auto C = np::ArrayCpu<TP>(rows, B.cols, 0);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, rows, B.cols, cols, 1.0, reinterpret_cast<float*> (mat), cols, reinterpret_cast<float*>(B.mat), B.cols, 0.0, reinterpret_cast<float*>(C.mat), B.cols);
        return C;
    }
    else if (std::is_same<TP, double>::value) {
        auto C = np::ArrayCpu<TP>(rows, B.cols, 0);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, rows, B.cols, cols, 1.0, reinterpret_cast<double*> (mat), cols, reinterpret_cast<double*>(B.mat), B.cols, 0.0, reinterpret_cast<double*>(C.mat), B.cols);
        return C;

    }
    return NULL;
}



template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator+(np::ArrayCpu<TP>& B) {
    //when A is scalaer
    
    if (this->rows == 1 and this->cols == 1) {
      
        auto C = ArrayCpu<TP>(B.rows, B.cols);
        kernelMatAddScaler<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
        return C;

    }
    else if (B.rows == 1 and B.cols == 1) {
        
        auto C = ArrayCpu<TP>(this->rows,this->cols);
        kernelMatAddScaler<TP>(this->mat, B(0, 0), C.mat, this->rows, this->cols);
        return C;
    }
    else if (this->rows == B.rows && this->cols == B.cols) {
       
      auto C = ArrayCpu<TP>(this->rows, this->cols);
      kernelMatAddMat<TP>(this->mat, B.mat, C.mat, rows, cols);
      return C;
  }
    else if (this->rows == 1 || this->cols == 1) {
        
      int vecDim = this->rows > this->cols ? this->rows : this->cols;
      //when the second matrix is square matrix then exception
      if (vecDim == B.rows && vecDim==B.cols) {
          
      }
      else if (vecDim == B.rows) {
          auto C = ArrayCpu<TP>(B.rows, B.cols);
          kernelMatAddVecAlongCols<TP>(B.mat, this->mat,C.mat, B.rows, B.cols);
          return C;
      }
      else if (vecDim == B.cols) {
          auto C = ArrayCpu<TP>(B.rows, B.cols);
          kernelMatAddVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
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
          kernelMatAddVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
          return C;
      }
      else if (vecDim == this->cols) {
          auto C = ArrayCpu<TP>(this->rows, this->cols);
          kernelMatAddVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
          return C;
      }
  }

}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator+(TP b) {
    auto C = np::ArrayCpu<TP>(this->rows,this->cols);
    kernelMatAddScaler<TP>(this->mat,b, C.mat, this->rows, this->cols);
    return C;

}

template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator-(np::ArrayCpu<TP>& B) {
    //when A is scalaer

    if (this->rows == 1 and this->cols == 1) {

        auto C = ArrayCpu<TP>(B.rows, B.cols);
        kernelMatSubScaler<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
        return C;

    }
    else if (B.rows == 1 and B.cols == 1) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatSubScaler<TP>(this->mat, B(0, 0), C.mat, this->rows, this->cols);
        return C;
    }
    else if (this->rows == B.rows && this->cols == B.cols) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatSubMat<TP>(this->mat, B.mat, C.mat, rows, cols);
        return C;
    }
    else if (this->rows == 1 || this->cols == 1) {

        int vecDim = this->rows > this->cols ? this->rows : this->cols;
        //when the second matrix is square matrix then exception
        if (vecDim == B.rows && vecDim == B.cols) {

        }
        else if (vecDim == B.rows) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatSubVecAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
            return C;
        }
        else if (vecDim == B.cols) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatSubVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
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
            kernelMatSubVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
        else if (vecDim == this->cols) {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatSubVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
    }

}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator-(TP b) {
    auto C = np::ArrayCpu<TP>(this->rows, this->cols);
    kernelMatSubScaler<TP>(this->mat, b, C.mat, this->rows, this->cols);
    return C;

}
//--

template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator*(np::ArrayCpu<TP>& B) {
    //when A is scalaer

    if (this->rows == 1 and this->cols == 1) {

        auto C = ArrayCpu<TP>(B.rows, B.cols);
        kernelMatMulScaler<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
        return C;

    }
    else if (B.rows == 1 and B.cols == 1) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatMulScaler<TP>(this->mat, B(0, 0), C.mat, this->rows, this->cols);
        return C;
    }
    else if (this->rows == B.rows && this->cols == B.cols) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatMulMat<TP>(this->mat, B.mat, C.mat, rows, cols);
        return C;
    }
    else if (this->rows == 1 || this->cols == 1) {

        int vecDim = this->rows > this->cols ? this->rows : this->cols;
        //when the second matrix is square matrix then exception
        if (vecDim == B.rows && vecDim == B.cols) {

        }
        else if (vecDim == B.rows) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatMulVecAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
            return C;
        }
        else if (vecDim == B.cols) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatMulVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
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
            kernelMatMulVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
        else if (vecDim == this->cols) {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatMulVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
    }

}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator*(TP b) {
    auto C = np::ArrayCpu<TP>(this->rows, this->cols);
    kernelMatMulScaler<TP>(this->mat, b, C.mat, this->rows, this->cols);
    return C;

}

template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator/(np::ArrayCpu<TP>& B) {
    //when A is scalaer

    if (this->rows == 1 and this->cols == 1) {

        auto C = ArrayCpu<TP>(B.rows, B.cols);
        kernelMatDivScaler<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
        return C;

    }
    else if (B.rows == 1 and B.cols == 1) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatDivScaler<TP>(this->mat, B(0, 0), C.mat, this->rows, this->cols);
        return C;
    }
    else if (this->rows == B.rows && this->cols == B.cols) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatDivMat<TP>(this->mat, B.mat, C.mat, rows, cols);
        return C;
    }
    else if (this->rows == 1 || this->cols == 1) {

        int vecDim = this->rows > this->cols ? this->rows : this->cols;
        //when the second matrix is square matrix then exception
        if (vecDim == B.rows && vecDim == B.cols) {

        }
        else if (vecDim == B.rows) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatDivVecAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
            return C;
        }
        else if (vecDim == B.cols) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatDivVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
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
            kernelMatDivVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
        else if (vecDim == this->cols) {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatDivVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
    }

}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator/(TP b) {
    auto C = np::ArrayCpu<TP>(this->rows, this->cols);
    kernelMatDivScaler<TP>(this->mat, b, C.mat, this->rows, this->cols);
    return C;

}

//----

template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator>(np::ArrayCpu<TP>& B) {
    //when A is scalaer

    if (this->rows == 1 and this->cols == 1) {

        auto C = ArrayCpu<TP>(B.rows, B.cols);
        kernelMatGreaterThanScaler<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
        return C;

    }
    else if (B.rows == 1 and B.cols == 1) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatGreaterThanScaler<TP>(this->mat, B(0, 0), C.mat, this->rows, this->cols);
        return C;
    }
    else if (this->rows == B.rows && this->cols == B.cols) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatGreaterThanMat<TP>(this->mat, B.mat, C.mat, rows, cols);
        return C;
    }
    else if (this->rows == 1 || this->cols == 1) {

        int vecDim = this->rows > this->cols ? this->rows : this->cols;
        //when the second matrix is square matrix then exception
        if (vecDim == B.rows && vecDim == B.cols) {

        }
        else if (vecDim == B.rows) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatGreaterThanVecAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
            return C;
        }
        else if (vecDim == B.cols) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatGreaterThanVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
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
            kernelMatGreaterThanVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
        else if (vecDim == this->cols) {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatGreaterThanVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
    }

}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator>(TP b) {
    auto C = np::ArrayCpu<TP>(this->rows, this->cols);
    kernelMatGreaterThanScaler<TP>(this->mat, b, C.mat, this->rows, this->cols);
    return C;

}

//-----------

template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator<(np::ArrayCpu<TP>& B) {
    //when A is scalaer

    if (this->rows == 1 and this->cols == 1) {

        auto C = ArrayCpu<TP>(B.rows, B.cols);
        kernelMatLessThanScaler<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
        return C;

    }
    else if (B.rows == 1 and B.cols == 1) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatLessThanScaler<TP>(this->mat, B(0, 0), C.mat, this->rows, this->cols);
        return C;
    }
    else if (this->rows == B.rows && this->cols == B.cols) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatLessThanMat<TP>(this->mat, B.mat, C.mat, rows, cols);
        return C;
    }
    else if (this->rows == 1 || this->cols == 1) {

        int vecDim = this->rows > this->cols ? this->rows : this->cols;
        //when the second matrix is square matrix then exception
        if (vecDim == B.rows && vecDim == B.cols) {

        }
        else if (vecDim == B.rows) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatLessThanVecAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
            return C;
        }
        else if (vecDim == B.cols) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatLessThanVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
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
            kernelMatLessThanVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
        else if (vecDim == this->cols) {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatLessThanVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
    }

}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator<(TP b) {
    auto C = np::ArrayCpu<TP>(this->rows, this->cols);
    kernelMatLessThanScaler<TP>(this->mat, b, C.mat, this->rows, this->cols);
    return C;

}

//-----------------------

template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator>=(np::ArrayCpu<TP>& B) {
    //when A is scalaer

    if (this->rows == 1 and this->cols == 1) {

        auto C = ArrayCpu<TP>(B.rows, B.cols);
        kernelMatGreaterThanEqualScaler<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
        return C;

    }
    else if (B.rows == 1 and B.cols == 1) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatGreaterThanEqualScaler<TP>(this->mat, B(0, 0), C.mat, this->rows, this->cols);
        return C;
    }
    else if (this->rows == B.rows && this->cols == B.cols) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatGreaterThanEqualMat<TP>(this->mat, B.mat, C.mat, rows, cols);
        return C;
    }
    else if (this->rows == 1 || this->cols == 1) {

        int vecDim = this->rows > this->cols ? this->rows : this->cols;
        //when the second matrix is square matrix then exception
        if (vecDim == B.rows && vecDim == B.cols) {

        }
        else if (vecDim == B.rows) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatGreaterThanEqualVecAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
            return C;
        }
        else if (vecDim == B.cols) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatGreaterThanEqualVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
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
            kernelMatGreaterThanEqualVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
        else if (vecDim == this->cols) {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatGreaterThanEqualVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
    }

}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator>=(TP b) {
    auto C = np::ArrayCpu<TP>(this->rows, this->cols);
    kernelMatGreaterThanEqualScaler<TP>(this->mat, b, C.mat, this->rows, this->cols);
    return C;

}



template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator<=(np::ArrayCpu<TP>& B) {
    //when A is scalaer

    if (this->rows == 1 and this->cols == 1) {

        auto C = ArrayCpu<TP>(B.rows, B.cols);
        kernelMatLessThanEqualScaler<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
        return C;

    }
    else if (B.rows == 1 and B.cols == 1) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatLessThanEqualScaler<TP>(this->mat, B(0, 0), C.mat, this->rows, this->cols);
        return C;
    }
    else if (this->rows == B.rows && this->cols == B.cols) {

        auto C = ArrayCpu<TP>(this->rows, this->cols);
        kernelMatLessThanEqualMat<TP>(this->mat, B.mat, C.mat, rows, cols);
        return C;
    }
    else if (this->rows == 1 || this->cols == 1) {

        int vecDim = this->rows > this->cols ? this->rows : this->cols;
        //when the second matrix is square matrix then exception
        if (vecDim == B.rows && vecDim == B.cols) {

        }
        else if (vecDim == B.rows) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatLessThanEqualVecAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
            return C;
        }
        else if (vecDim == B.cols) {
            auto C = ArrayCpu<TP>(B.rows, B.cols);
            kernelMatLessThanEqualVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
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
            kernelMatLessThanEqualVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
        else if (vecDim == this->cols) {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatLessThanEqualVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
            return C;
        }
    }

}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::operator<=(TP b) {
    auto C = np::ArrayCpu<TP>(this->rows, this->cols);
    kernelMatLessThanEqualScaler<TP>(this->mat, b, C.mat, this->rows, this->cols);
    return C;

}



template<typename TP>
void np::ArrayCpu<TP>::reshape(int rows, int cols) {
    if (rows * cols != this->rows * this->cols) {
        this->rows = cols;
        this->cols = rows;
    }


}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::T() {
    np::ArrayCpu<TP> matT = np::ArrayCpu<TP>(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matT.mat[j * rows + i] = this->mat[i * cols + j];
        }
    }
    return matT;

}


template<typename TP>
int np::ArrayCpu<TP>::at(int i, int j) {

    return mat[i * cols + j];
}


template<typename TP>
TP& np::ArrayCpu<TP>::operator()(int i, int j) {
    return mat[i * cols + j];
}


template<typename TP>
void np::ArrayCpu<TP>::set(int i, int j, int val) {
    mat[i * cols + j] = val;
}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::sum(int axis) {
    if (axis == 0) {
        np::ArrayCpu<TP> SumCol = np::ArrayCpu<TP>(1, cols, 0);
#pragma omp parallel for
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                SumCol(0, j) += this->mat[i * cols + j];
            }
        }
        return SumCol;
    }
    else if (axis == 1) {
        auto SumRow = np::ArrayCpu<TP>(rows, 1, 0);
#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            TP ans = 0;
            for (int j = 0; j < cols; j++) {
                ans += (*this)(i, j);
            }
            SumRow(i, 0) = ans;
        }
        return SumRow;
    }
    else {
        auto SumRow = np::ArrayCpu<TP>(1, rows, 0);
#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            TP ans = 0;
            for (int j = 0; j < cols; j++) {
                ans += (*this)(i, j);
            }
            SumRow(0, i) = ans;
        }
        TP ans = 0;
#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            ans += SumRow(0, i);
        }
        return np::ArrayCpu<TP>(1, 1, ans);

    }
}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::max(int axis) {
    if (axis == 0) {
        np::ArrayCpu<TP> SumCol = np::ArrayCpu<TP>(1, cols, 0);

#pragma omp parallel for
        for (int j = 0; j < cols; j++) {
            TP k = (*this)(0, j);
            for (int i = 1; i < rows; i++) {
                if ((*this)(i, j) > k) k = (*this)(i, j);
            }
            SumCol(0, j) = k;
        }
        return SumCol;
    }
    else if (axis == 1) {
        auto SumRow = np::ArrayCpu<TP>(rows, 1, 0);
#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            TP ans = (*this)(i, 0);
            for (int j = 1; j < cols; j++) {
                if ((*this)(i, j) > ans) ans = (*this)(i, j);
            }
            SumRow(i, 0) = ans;
        }
        return SumRow;
    }
    else {
        auto SumRow = np::ArrayCpu<TP>(1, rows, 0);
#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            TP ans = (*this)(i, 0);
            for (int j = 1; j < cols; j++) {
                if ((*this)(i, j) > ans) ans = (*this)(i, j);
            }
            SumRow(0, i) = ans;
        }
        TP ans = SumRow(0, 0);

        for (int i = 1; i < rows; i++) {
            if (SumRow(0, i) > ans) ans = SumRow(0, i);
        }
        return np::ArrayCpu<TP>(1, 1, ans);

    }
}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::min(int axis) {
    if (axis == 0) {
        np::ArrayCpu<TP> SumCol = np::ArrayCpu<TP>(1, cols, 0);

#pragma omp parallel for
        for (int j = 0; j < cols; j++) {
            TP k = (*this)(0, j);
            for (int i = 1; i < rows; i++) {
                if ((*this)(i, j) < k) k = (*this)(i, j);
            }
            SumCol(0, j) = k;
        }
        return SumCol;
    }
    else if (axis == 1) {
        auto SumRow = np::ArrayCpu<TP>(rows, 1, 0);
#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            TP ans = (*this)(i, 0);
            for (int j = 1; j < cols; j++) {
                if ((*this)(i, j) < ans) ans = (*this)(i, j);
            }
            SumRow(i, 0) = ans;
        }
        return SumRow;
    }
    else {
        auto SumRow = np::ArrayCpu<TP>(1, rows, 0);
#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            TP ans = (*this)(i, 0);
            for (int j = 1; j < cols; j++) {
                if ((*this)(i, j) < ans) ans = (*this)(i, j);
            }
            SumRow(0, i) = ans;
        }
        TP ans = SumRow(0, 0);

        for (int i = 1; i < rows; i++) {
            if (SumRow(0, i) < ans) ans = SumRow(0, i);
        }
        return np::ArrayCpu<TP>(1, 1, ans);

    }
}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::argmin(int axis) {
    if (axis == 0) {
        np::ArrayCpu<TP> SumCol = np::ArrayCpu<TP>(1, cols, 0);

#pragma omp parallel for
        for (int j = 0; j < cols; j++) {
            TP ind = 0;
            for (int i = 1; i < rows; i++) {
                if ((*this)(i, j) < (*this)(ind, j)) ind = i;
            }
            SumCol(0, j) = ind;
        }
        return SumCol;
    }
    else if (axis == 1) {
        auto SumRow = np::ArrayCpu<TP>(rows, 1, 0);
#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            TP ind = 0;
            for (int j = 1; j < cols; j++) {
                if ((*this)(i, j) < (*this)(i, ind)) ind = j;
            }
            SumRow(i, 0) = ind;
        }
        return SumRow;
    }
    else {
        auto SumRow = np::ArrayCpu<TP>(1, rows, 0);
#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            TP ind = 0;
            for (int j = 1; j < cols; j++) {
                if ((*this)(i, j) < (*this)(i, ind)) ind = j;
            }
            SumRow(0, i) = ind;
        }
        TP ans = 0;

        for (int i = 1; i < rows; i++) {
            if (SumRow(0, i) < SumRow(0, ans)) ans = i;
        }
        return np::ArrayCpu<TP>(1, 1, ans);

    }
}


template<typename TP>
np::ArrayCpu<TP> np::ArrayCpu<TP>::argmax(int axis) {
    if (axis == 0) {
        np::ArrayCpu<TP> SumCol = np::ArrayCpu<TP>(1, cols, 0);

#pragma omp parallel for
        for (int j = 0; j < cols; j++) {
            TP ind = 0;
            for (int i = 1; i < rows; i++) {
                if ((*this)(i, j) > (*this)(ind, j)) ind = i;
            }
            SumCol(0, j) = ind;
        }
        return SumCol;
    }
    else if (axis == 1) {
        auto SumRow = np::ArrayCpu<TP>(rows, 1, 0);
#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            TP ind = 0;
            for (int j = 1; j < cols; j++) {
                if ((*this)(i, j) > (*this)(i, ind)) ind = j;
            }
            SumRow(i, 0) = ind;
        }
        return SumRow;
    }
    else {
        auto SumRow = np::ArrayCpu<TP>(1, rows, 0);
#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            TP ind = 0;
            for (int j = 1; j < cols; j++) {
                if ((*this)(i, j) > (*this)(i, ind)) ind = j;
            }
            SumRow(0, i) = ind;
        }
        TP ans = 0;

        for (int i = 1; i < rows; i++) {
            if (SumRow(0, i) > SumRow(0, ans)) ans = i;
        }
        return np::ArrayCpu<TP>(1, 1, ans);

    }
}


template<typename TP>
std::ostream& operator<<(std::ostream& out, np::ArrayCpu<TP>& Arr) {
    out << "dimensions: " << Arr.rows << " X " << Arr.cols << "\n";
    for (int i = 0; i < Arr.rows; i++) {
        for (int j = 0; j < Arr.cols; j++) {
            out << Arr.mat[i * Arr.cols + j] << " ";
        }
        out << "\n";
    }
    return out;
}








}










#endif // !