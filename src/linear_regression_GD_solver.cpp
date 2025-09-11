#include "linear_regression_GD_solver.hpp"
#include "matrix.hpp"
#include "vectr.hpp"
#include "sparse_matrix.hpp"
#include "sparse_vector.hpp"
#include "random_quantities.hpp"
#include "index_shuffler.hpp"
#include <iostream>
#include <cmath>

LinearRegressionGDSolver::LinearRegressionGDSolver(double learningRate, size_t numStochasticSamples, size_t maxNumIterations, double tolerance)
:GradientDescentSolver(maxNumIterations, tolerance),
GradientDescentData(numStochasticSamples, learningRate)
{
}

void LinearRegressionGDSolver::solve(const Matrix& X, const Vector& y)
{
    setData(X, y);
    GradientDescentSolver::solve(X, y);
}

void LinearRegressionGDSolver::evaluateIncrements()
{
    m_indexer.update();

    // error vector
    std::vector<double> err = {};
    for(size_t i = 0; i < m_numRows; i++)
    {
        size_t iActual = m_indexer.getIndex(i);
        double sum = m_bias - m_py->getData()[iActual];
        for(size_t j = 0; j < m_numColumns; j++)
        {
            sum += m_pX->getData()[iActual][j] * m_weights.getData()[j];
        }
        err.push_back(sum);
    }
    std::vector<double> dCdwVec = {};
    for(size_t i = 0; i < m_numColumns; i++)
    {
        double sum = 0;
        for(size_t j = 0; j < m_numRows; j++)
        {
            // sum += (X-transpose[i][indexer.getIndex(j)] times err[j])
            // but X-transpose[i][indexer.getIndex(j)] = X[indexer.getIndex(j)][i]
            sum += (m_pX->getData()[m_indexer.getIndex(j)][i] * err[j]);
        }
        dCdwVec.push_back(sum);
    }

    // Partial derivative of cost function w.r.t. weights: dCdw = Xsam-transpose * err / numRows
    // Xsam -> entire X or sampled X in case of stochastic gradient descent
    // Increment in weights for gradient descent: dCdw * negative-learning-rate
    m_weightIncrements = Vector(dCdwVec) * m_constMult;

    // Partial derivative of cost function w.r.t. bias: dCdb = SUM(err) / numRows
    // Increment in bias for gradiant descent: dCdb * negative-learning-rate
    m_biasIncrement = Vector(err).getSum() * m_constMult;
}