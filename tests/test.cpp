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

    LinearRegressionSolver linRegSolverAnalytical = LinearRegressionSolver();
    linRegSolverAnalytical.solve(X, y, LinearRegressionSolver::ANALYTICAL);
    Vector weightsAnalytical = linRegSolverAnalytical.getWeights();
    double biasAnalytical = linRegSolverAnalytical.getBias();

    LinearRegressionSolver linRegSolverGD = LinearRegressionSolver();
    linRegSolverGD.solve(X, y, LinearRegressionSolver::GRADIENT_DESCENT);
    Vector weightsGD = linRegSolverGD.getWeights();
    double biasGD = linRegSolverGD.getBias();

    ///*
    LinearRegressionSolver linRegSolverSGD = LinearRegressionSolver();
    linRegSolverSGD.solve(X, y, LinearRegressionSolver::STOCHASTIC_GRADIENT_DESCENT);
    Vector weightsSGD = linRegSolverSGD.getWeights();
    double biasSGD = linRegSolverSGD.getBias();//*/
    return 0;
}