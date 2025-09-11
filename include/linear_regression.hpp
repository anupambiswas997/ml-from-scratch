#ifndef LINEAR_REGRESSION_ANALYTICAL_SOLVER_HPP
#define LINEAR_REGRESSION_ANALYTICAL_SOLVER_HPP

#include "base_solver.hpp"

class Vector;

class LinearRegressionAnalyticalSolver: virtual public BaseSolver
{
public:
    LinearRegressionAnalyticalSolver();
    virtual void solve(const Matrix& X, const Vector& y);
};

#endif