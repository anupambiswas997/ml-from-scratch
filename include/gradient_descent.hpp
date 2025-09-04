#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include "vectr.hpp"

class IGradientDescentSolver
{
protected:
    Vector m_weights;
    double m_bias;
    Vector m_weightIncrements;
    double m_biasIncrement;
    size_t m_iterationCount;
    size_t m_maxIterations;
    double m_tolerance;
    double m_maxIncrement;
public:
    IGradientDescentSolver(size_t numIterations, double tolerance);
    virtual void evaluateIncrements() = 0;
    virtual bool shouldContinueIterating();
    virtual void log() const;
    virtual void solve(const Matrix& X, const Vector& y);
    const Vector& getWeights() const;
    double getBias() const;
};

#endif