#include "linear_regression.hpp"
#include <iostream>
#include "matrix.hpp"
#include "vectr.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    Matrix m = {};
    Vector v = {};
    LinearRegressionSolver linRegSolver = LinearRegressionSolver();
    linRegSolver.solve({}, {});
    return 0;
}