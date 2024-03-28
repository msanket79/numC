## NUMC

A C++ port of NumPy, currently supporting 1D and 2D arrays with OpenMP-accelerated functions.

**Purpose**

- NUMC aims to bring the familiar array-centric computing style of NumPy to C++ developers.
- It provides a performant way to work with arrays in C++, leveraging the power of OpenMP for parallelization.
- Use cases could include scientific computing, numerical analysis, and other tasks that benefit from efficient array manipulation.

**Installation**

- **Prerequisites:**
  - A C++ compiler with OpenMP support (e.g., GCC, Clang)
  - OpenBLAS library (link to installation instructions if needed)
- **Instructions:**
  1. Download or clone the NUMC project repository.
  2. Open the `NUMC.sln` solution file in Visual Studio.
  3. Ensure OpenBLAS is linked correctly in your project settings.
  4. Build the solution.

# ArrayCPU Class

The `ArrayCPU` class provides a template class for managing 1D and 2D arrays on the CPU. It supports basic arithmetic operations, element-wise comparison operations, and various array manipulation functions.

# Broadcasting

The `ArrayCPU` class supports broadcasting, allowing for element-wise operations between arrays of different shapes.

we can also add constants to the array

## Broadcasting Rules

When operating on two arrays, NumC compares their shapes element-wise. It comapres coumns first and then rows.

Two dimensions are compatible when

1- they are equal, or

2- one of them is 1.

If these conditions are not met, then error is returned

## axis

In our NumC library, the axis parameter mirrors NumPy's behavior.
Specifically,setting

axis=1 corresponds to operations along the x-axis,

axis=0 corresponds to operations along the y-axis,

axis=-1 implies operations considering all elements.

## Constructors

### `ArrayCPU(int rows=1, int cols=1, TP val=0)`

- **Description:** Constructs an `ArrayCPU` object with the specified number of rows and columns, initialized with the given value `val` default `val`=0.

## Member Functions

### `int at(int i, int j)`

- **Description:** Returns the element at the specified row `i` and column `j`.

### `void set(int i, int j, int val)`

- **Description:** Sets the element at the specified row `i` and column `j` to the given value `val`.

### `arr(i,j)`

- **Description:** Allows accessing elements of the array using the function call syntax, e.g., `arr(i, j)`.

### `Operator+ Overload`

The `operator+` overload allows adding two Arrays, element-wise `ex` A+B.

It also supports broadcasting (so we can add constants also and arrays of different compatible shapes)

### `ArrayCPU<TP> T()`

- **Description:** Returns the transpose of the array.

### `ArrayCPU<TP> sum(int axis=-1)`

- **Description:** Computes the sum of elements along the specified axis.

### `ArrayCPU<TP> min(int axis=-1)`

- **Description:** Finds the minimum elements along the specified axis.

### `ArrayCPU<TP> max(int axis=-1)`

- **Description:** Finds the maximum elements along the specified axis.

### `ArrayCPU<TP> argmin(int axis=-1)`

- **Description:** Finds the indices of the minimum elements along the specified axis.

### `ArrayCPU<TP> argmax(int axis=-1)`

- **Description:** Finds the indices of the maximum elements along the specified axis.

### `ArrayCPU<TP> dot(ArrayCPU<TP>& B)`

- **Description:** Computes dot product with another array B. returns `A dot B`

  A represents our array object

  B represents the input array array object

### `ArrayCPU<TP> dotT(ArrayCPU<TP>& B)`

- **Description:** Computes dot product with tranpose of B. returns `A dot (B.T)`

  A represents our array object

  B represents the input array array object

### `ArrayCPU<TP> dot(ArrayCPU<TP>& B)`

- **Description:** Computes dot of the transpose of A with B. returns `(A.T) dot B`

  A represents our array object

  B represents the input array array object

### `void reshape(int rows, int cols)`

- **Description:** Reshapes the array to the specified number of rows and columns.

The product of dimensions of our current array should rows\*cols otherwise error will be thrown

## Operator Overloads

The `ArrayCPU` class overloads several arithmetic and comparison operators, allowing for intuitive element-wise operations with scalars and other arrays.

For example, you can use the `+` operator to add two arrays element-wise:

````cpp
ArrayCPU<int> A(2, 2, 1);
ArrayCPU<int> B(2, 2, 2);
ArrayCPU<int> C = A + B;


**Basic Usage Example**

```c++
#include "ArrayCPU.h" // Assuming your ArrayCPU class is in a header file

int main() {
    ArrayCPU<int> arr1(2, 3, 1);  // 2x3 array filled with 1s
    ArrayCPU<int> arr2(2, 3);

    // Fill arr2 with values
    arr2 << 1, 2, 3, 4, 5, 6;

    ArrayCPU<int> result = arr1 + arr2;
    std::cout << result << std::endl;  // Prints the result array
}
````
