#include "logistic_regression_solver.hpp"

LogisticRegressionSolver::LogisticRegressionSolver(double learningRate, size_t numStochasticSamples, size_t maxNumIterations, double tolerance)
:IGradientDescentSolver(maxNumIterations, tolerance)
{
    //
}

void LogisticRegressionSolver::evaluateIncrements()
{
    //
}

void LogisticRegressionSolver::solve(const Matrix& X, const Vector& y)
{
    //
}