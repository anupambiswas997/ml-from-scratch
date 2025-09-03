#include "linear_regression.hpp"
#include <iostream>
#include "matrix.hpp"
#include "sparse_matrix.hpp"
#include "sparse_vector.hpp"
#include "random_quantities.hpp"
#include <cmath>
#include "gradient_descent.hpp"
#include "index_shuffler.hpp"

LinearRegressionGDSolver::LinearRegressionGDSolver(size_t numRows, size_t numColumns, double learningRate, size_t numStochasticSamples, size_t maxNumIterations, double tolerance)
:IGradientDescentSolver(maxNumIterations, tolerance),
m_indexer(numRows, (numStochasticSamples > 0)),
m_learningRate(learningRate),
m_isStochastic(numStochasticSamples > 0)
{
    // add assert later: numStochasticSamples < X.getNumRows()
    m_numRows = (numStochasticSamples > 0) ? numStochasticSamples : numRows;
    m_numColumns = numColumns;
    m_mInv = 1.0 / m_numRows;
    m_mInvNegLR = -m_mInv * m_learningRate;
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
    m_weightIncrements = Vector(dCdwVec) * m_mInvNegLR;

    // Partial derivative of cost function w.r.t. bias: dCdb = SUM(err) / numRows
    // Increment in bias for gradiant descent: dCdb * negative-learning-rate
    m_biasIncrement = Vector(err).getSum() * m_mInvNegLR;
}

LinearRegressionSolver::LinearRegressionSolver()
{
    std::cout << "Created LinearRegressionSolver object" << std::endl;
}

void LinearRegressionSolver::solve(const Matrix& X, const Vector& y)
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

const Vector& LinearRegressionSolver::getWeights() const
{
    return m_weights;
}

double LinearRegressionSolver::getBias() const
{
    return m_bias;
}