#include "linear_regression.hpp"
#include <iostream>
#include "matrix.hpp"
#include "vectr.hpp"
#include "random_quantities.hpp"
#include "test_utils.hpp"

using namespace std;

void testLinearRegression(size_t sampleSize=1000, size_t numFeatures=1)
{
    Matrix X = getRandomMatrix(sampleSize, numFeatures, -3, 3);
    Vector weights = getRandomVector(numFeatures, -2, 2);
    double bias = getRandom();
    Vector noise = getRandomVector(sampleSize, -0.2, 0.2);
    Vector y = ((X * weights) + bias) + noise;

    LinearRegressionSolver linRegSolverAnalytical = LinearRegressionSolver();
    SHOW_TIME_ELAPSED("\nLINEAR REGRESSION - ANALYTICAL", linRegSolverAnalytical.solve(X, y, LinearRegressionSolver::ANALYTICAL));
    Vector weightsAnalytical = linRegSolverAnalytical.getWeights();
    double biasAnalytical = linRegSolverAnalytical.getBias();

    LinearRegressionSolver linRegSolverGD = LinearRegressionSolver();
    SHOW_TIME_ELAPSED("\nLINEAR REGRESSION - GRADIENT DESCENT", linRegSolverGD.solve(X, y, LinearRegressionSolver::BATCH_GRADIENT_DESCENT));
    Vector weightsGD = linRegSolverGD.getWeights();
    double biasGD = linRegSolverGD.getBias();

    ///*
    LinearRegressionSolver linRegSolverSGD = LinearRegressionSolver();
    SHOW_TIME_ELAPSED("\nLINEAR REGRESSION - STOCHASTIC GRADIENT DESCENT", linRegSolverSGD.solve(X, y, LinearRegressionSolver::STOCHASTIC_GRADIENT_DESCENT));
    Vector weightsSGD = linRegSolverSGD.getWeights();
    double biasSGD = linRegSolverSGD.getBias();

    std::cout << "WEIGHTS:" << std::endl;
    std::cout << "actual:" << std::endl << weights.getText() << std::endl;
    std::cout << "analytical:" << std::endl << weightsAnalytical.getText() << std::endl;
    std::cout << "gradient descent:" << std::endl << weightsGD.getText() << std::endl;
    std::cout << "stochastic gradient descent:" << std::endl << weightsSGD.getText() << std::endl;

    std::cout << "BIAS:" << std::endl;
    std::cout << "actual:" << std::endl << bias << std::endl;
    std::cout << "analytical:" << std::endl << biasAnalytical << std::endl;
    std::cout << "gradient descent:" << std::endl << biasGD << std::endl;
    std::cout << "stochastic gradient descent:" << std::endl << biasSGD << std::endl;
    //*/
}

int main(int argc, char *argv[])
{
    testLinearRegression();
    return 0;
}