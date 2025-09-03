#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "vectr.hpp"
#include "gradient_descent.hpp"
#include "index_shuffler.hpp"

class LinearRegressionGDSolver: virtual public IGradientDescentSolver
{
    double m_learningRate;
    size_t m_numRows;
    size_t m_numColumns;
    bool m_isStochastic;
    IndexShuffler m_indexer;
    double m_mInv;
    double m_mInvNegLR;
public:
    LinearRegressionGDSolver(size_t numRows, size_t numColumns, double learningRate=1e-4, size_t numStochasticSamples=0, size_t maxNumIterations=100000, double tolerance=1e-8);
    virtual void evaluateIncrements();
};

class LinearRegressionSolver
{
    Vector m_weights;
    double m_bias;
public:
    LinearRegressionSolver();
    void solve(const Matrix& X, const Vector& y);
    const Vector& getWeights() const;
    double getBias() const;
};

#endif