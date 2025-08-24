#include "linear_regression.hpp"
#include <iostream>
#include "matrix.hpp"
#include "sparse_matrix.hpp"
#include "sparse_vector.hpp"
#include "random_quantities.hpp"
#include <cmath>

LinearRegressionSolver::LinearRegressionSolver()
{
    std::cout << "Created LinearRegression object" << std::endl;
}

void LinearRegressionSolver::solveAnalytical(const Matrix& X, const Vector& y)
{
    size_t m = X.getNumRows();
    double oneByM = 1.0 / m;
    Matrix XT = X.getTranspose();
    SparseMatrix iden(0, m, m);
    for(size_t i = 0; i < m; i++)
    {
        iden[i][i] = 1;
    }
    SparseMatrix U = iden - oneByM;
    Matrix XTU = XT * U;
    Matrix XTUX = XTU * X;
    Matrix XTUXinverse = XTUX.getInverse();
    Vector XTUy = XTU * y;
    m_weights = XTUXinverse * XTUy;
    double sum = 0;
    Vector Xw = X * m_weights;
    for(size_t i = 0; i < m; i++)
    {
        sum += (y[i] - Xw[i]);
    }
    m_bias = oneByM * sum;
}

void LinearRegressionSolver::solveGradientDescent(const Matrix& X, const Vector& y)
{
    m_bias = getRandom();
    m_weights = getRandomVector(X.getNumColumns());
    bool cond = true;
    size_t iterationCount = 0;
    double error;
    double minv = 1.0 / X.getNumRows();
    size_t maxNumIterations = 10000; // to be parameterized
    double tolerance = 1.0e-8; // to be parameterized
    double learningRate = 1.0e-4; // to be parameterized
    double negLearningRate = -learningRate;

    Matrix XT = X.getTranspose();
    bool errorCond, iterCond;

    while(cond)
    {
        // error vector
        Vector errVec = (X * m_weights + m_bias) - y;
        // Partial derivative of cost-function C with respect to weights
        Vector dCdw = (XT * errVec) * minv;
        // Partial derivative of cost-function C with respect to bias
        double dCdb = errVec.getSum() * minv;
        Vector weightsIncrement = dCdw * negLearningRate;
        double biasIncrement = dCdb * negLearningRate;
        m_weights = m_weights + weightsIncrement;
        m_bias += biasIncrement;

        // Compute maximum error
        double absMinWError = fabs(weightsIncrement.getMin());
        double absMaxWError = fabs(weightsIncrement.getMax());
        double wError = (absMinWError < absMaxWError) ? absMinWError : absMaxWError;
        double absBiasError = fabs(biasIncrement);
        error = (wError < absBiasError) ? absBiasError : wError;

        iterationCount++;
        errorCond = error > tolerance;
        iterCond = iterationCount < maxNumIterations;
        // The looping to find optimal weights continues if:
        //   a. the maximum error is still greater than the provided tolerance, and
        //   b. the number of iterations is smaller than the provided iteration limit.
        cond = errorCond && iterCond;
    }
    std::cout << "Linear regression gradient descent finished with:" << std::endl;
    std::cout << "Number of iterations completed: " << iterationCount << std::endl;
    std::cout << "Maximum number of iterations: " << maxNumIterations << std::endl;
    std::cout << "Last absolute weights/bias increment: " << error << std::endl;
    std::cout << "Error tolerance used: " << tolerance << std::endl;
}

void LinearRegressionSolver::solveStochasticGradientDescent(const Matrix& X, const Vector& y)
{
}

void LinearRegressionSolver::solve(const Matrix& X, const Vector& y, int solutionMethod)
{
    if(solutionMethod == ANALYTICAL)
    {
        solveAnalytical(X, y);
    }
    else if(solutionMethod == GRADIENT_DESCENT)
    {
        solveGradientDescent(X, y);
    }
    else if(solutionMethod == STOCHASTIC_GRADIENT_DESCENT)
    {
        solveStochasticGradientDescent(X, y);
    }
    else
    {
        std::cout << "*** Incorrect solution method: " << solutionMethod << std::endl;
    }
}

const Vector& LinearRegressionSolver::getWeights() const
{
    return m_weights;
}

double LinearRegressionSolver::getBias() const
{
    return m_bias;
}