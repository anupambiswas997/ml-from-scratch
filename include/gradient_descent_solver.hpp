#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include "vectr.hpp"
#include "simple_solver.hpp"

class GradientDescentSolver: virtual public SimpleSolver
{
protected:
    Vector m_weightIncrements;
    double m_biasIncrement;
    size_t m_iterationCount;
    size_t m_maxIterations;
    double m_tolerance;
    double m_maxIncrement;
public:
    GradientDescentSolver(size_t numIterations, double tolerance);
    virtual void evaluateIncrements() = 0;
    virtual bool shouldContinueIterating();
    virtual void log() const;
    virtual void solve(const Matrix& X, const Vector& y);
};

#endif