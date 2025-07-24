#include "linear_regression.hpp"
#include <iostream>
#include "matrix.hpp"
#include "vectr.hpp"
#include "random_quantities.hpp"

using namespace std;

struct TestData
{
    Matrix X;
    Vector y;
    Vector w;
    double b;
    TestData(const Matrix& _X, const Vector& _y, const Vector& _w, double _b)
    {
        X = _X;
        y = _y;
        w = _w;
        b = _b;
    };
};

TestData getDataSet(size_t sampleSize=1000, size_t numFeatures=1)
{
    Matrix X = getRandomMatrix(sampleSize, numFeatures, -3, 3);
    Vector w = getRandomVector(numFeatures, -2, 2);
    double b = getRandom();
    Vector noise = getRandomVector(sampleSize, -0.2, 0.2);
    Vector y = ((X * w) + b) + noise;
    return TestData(X, y, w, b);
}

int main(int argc, char *argv[])
{
    TestData tData = getDataSet();
    Matrix& X = tData.X;
    Vector& y = tData.y;
    LinearRegressionSolver linRegSolver = LinearRegressionSolver();
    linRegSolver.solve(X, y);
    return 0;
}