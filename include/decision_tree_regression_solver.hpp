#ifndef DECISION_TREE_REGRESSION_SOLVER_HPP
#define DECISION_TREE_REGRESSION_SOLVER_HPP

#include "base_solver.hpp"

// DecisionTreeRegressionSolver is the class that can be used
// to create a decision tree for a regression problem.

class DecisionTreeRegressionSolver: virtual public BaseSolver
{
public:
    virtual void solve(const Matrix& X, const Vector& y);
    virtual Vector predict(const Matrix& X) const;
    virtual double predict(const Vector& xrow) const;
};

#endif