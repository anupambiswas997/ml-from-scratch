#ifndef SIMPLE_SOLVER_HPP
#define SIMPLE_SOLVER_HPP

#include "vectr.hpp"
#include "matrix.hpp"

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
    virtual Vector predict(const Matrix& X) const
    {
        return (X * m_weights) + m_bias;
    }
    virtual double predict(const Vector& y) const
    {
        return y.dot(m_weights) + m_bias;
    }
};

#endif