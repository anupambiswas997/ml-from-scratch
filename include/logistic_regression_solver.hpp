#ifndef LOGISTIC_REGRESSION_SOLVER_HPP
#define LOGISTIC_REGRESSION_SOLVER_HPP

#include "vectr.hpp"
#include "gradient_descent.hpp"
#include "gradient_descent_data.hpp"

class LogisticRegressionSolver:
    virtual public IGradientDescentSolver,
    virtual public GradientDescentData
{
public:
    LogisticRegressionSolver(double learningRate=1e-4, size_t numStochasticSamples=0, size_t maxNumIterations=100000, double tolerance=1e-8);
    virtual void evaluateIncrements();
    virtual void solve(const Matrix& X, const Vector& y);
    Vector getProbability(const Matrix& X) const;
    double getProbability(const Vector& xrow) const;
    virtual Vector predict(const Matrix& X) const;
    virtual double predict(const Vector& xrow) const;
};

#endif