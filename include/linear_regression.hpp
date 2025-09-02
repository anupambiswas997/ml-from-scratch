#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "vectr.hpp"
#include "gradient_descent.hpp"
#include "index_shuffler.hpp"

class Indexer
{
    std::vector<size_t> m_indices;
    bool m_doShuffle;
public:
    Indexer(size_t size, bool doShuffle);
    void update();
    size_t getIndex(size_t i);
};

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
    LinearRegressionGDSolver(const Matrix& X, const Vector& y, double learningRate=1e-4, size_t numStochasticSamples=0, size_t maxNumIterations=100000, double tolerance=1e-8);
    virtual void evaluateIncrements();
};

class LinearRegressionSolver
{
    Vector m_weights;
    double m_bias;
public:
    enum SolutionMethodEnum
    {
        ANALYTICAL,                  // for analytical solution using linear algebra operations
        BATCH_GRADIENT_DESCENT,      // for gradient descent method using entire dataset
        STOCHASTIC_GRADIENT_DESCENT  // for gradient descent using a subset of dataset
    };
    LinearRegressionSolver();
    void solve(const Matrix& X, const Vector& y, int solutionMethod=SolutionMethodEnum::ANALYTICAL);
    void solveAnalytical(const Matrix& X, const Vector& y);
    void solveGradientDescent(const Matrix& X, const Vector& y, size_t numStochasticSamples=0, double learningRate=1.0e-4, double tolerance=1.0e-8, size_t maxNumIterations=100000);
    const Vector& getWeights() const;
    double getBias() const;
};

#endif