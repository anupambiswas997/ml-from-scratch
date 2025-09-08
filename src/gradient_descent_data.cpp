#include "gradient_descent_data.hpp"
#include <cassert>

GradientDescentData::GradientDescentData(size_t numStochasticSamples, double learningRate)
{
    m_numStochasticSamples = numStochasticSamples;
    m_learningRate = learningRate;
}

void GradientDescentData::setData(const Matrix& X, const Vector& y)
{
    assert(m_numStochasticSamples < X.getNumRows());
    bool isStochasticGD = (m_numStochasticSamples > 0);
    m_pX = &X;
    m_py = &y;
    m_numRows = isStochasticGD ? m_numStochasticSamples : X.getNumRows();
    m_numColumns = X.getNumColumns();
    m_indexer = IndexShuffler(X.getNumRows(), isStochasticGD);
    m_constMult = -m_learningRate / (1.0 * m_numRows);
}