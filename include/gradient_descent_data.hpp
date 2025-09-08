#ifndef GRADIENT_DESCENT_DATA_HPP
#define GRADIENT_DESCENT_DATA_HPP

#include "matrix.hpp"
#include "vectr.hpp"
#include "index_shuffler.hpp"

class GradientDescentData
{
protected:
    const Matrix* m_pX;
    const Vector* m_py;
    double m_learningRate;
    size_t m_numRows;
    size_t m_numColumns;
    size_t m_numStochasticSamples;
    IndexShuffler m_indexer;
    double m_constMult;
public:
    GradientDescentData(size_t numStochasticSamples, double learningRate);
    virtual void setData(const Matrix& X, const Vector& y);
};

#endif