#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "vectr.hpp"
#include "gradient_descent_data.hpp"
#include "gradient_descent_solver.hpp"
#include "index_shuffler.hpp"
#include "base_solver.hpp"

class LinearRegressionGDSolver:
    virtual public GradientDescentSolver,
    virtual public GradientDescentData
{
public:
    LinearRegressionGDSolver(double learningRate=1e-4, size_t numStochasticSamples=0, size_t maxNumIterations=100000, double tolerance=1e-8);
    virtual void evaluateIncrements();
    virtual void solve(const Matrix& X, const Vector& y);
};

class LinearRegressionSolver: virtual public BaseSolver
{
public:
    LinearRegressionSolver();
    virtual void solve(const Matrix& X, const Vector& y);
};

#endif