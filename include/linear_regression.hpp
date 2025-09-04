#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "vectr.hpp"
#include "gradient_descent.hpp"
#include "index_shuffler.hpp"

class LinearRegressionGDSolver: virtual public IGradientDescentSolver
{
    const Matrix* m_pX;
    const Vector* m_py;
    double m_learningRate;
    size_t m_numRows;
    size_t m_numColumns;
    size_t m_numStochasticSamples;
    IndexShuffler m_indexer;
    double m_mInv;
    double m_mInvNegLR;
public:
    LinearRegressionGDSolver(double learningRate=1e-4, size_t numStochasticSamples=0, size_t maxNumIterations=100000, double tolerance=1e-8);
    virtual void evaluateIncrements();
    virtual void solve(const Matrix& X, const Vector& y);
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