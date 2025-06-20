#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "common_types.hpp"

class LinearRegressionSolver
{
    Vector m_weights;
    double m_bias;
public:
    LinearRegressionSolver();
    void solve(const Matrix& X, const Vector& y);
};

#endif