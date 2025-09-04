#ifndef SIMPLE_SOLVER_HPP
#define SIMPLE_SOLVER_HPP

#include "vectr.hpp"

class SimpleSolver
{
protected:
    Vector m_weights;
    double m_bias;
public:
    virtual const Vector& getWeights() const
    {
        return m_weights;
    }
    virtual double getBias() const
    {
        return m_bias;
    }
    virtual void solve(const Matrix& X, const Vector& y) = 0;
};

#endif