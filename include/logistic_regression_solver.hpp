#ifndef LOGISTIC_REGRESSION_SOLVER_HPP
#define LOGISTIC_REGRESSION_SOLVER_HPP

#include "vectr.hpp"
#include "gradient_descent_solver.hpp"
#include "gradient_descent_data.hpp"

class LogisticRegressionSolver:
    virtual public IGradientDescentSolver,
    virtual public GradientDescentData
{
public:
    LogisticRegressionSolver(double learningRate=1e-4, size_t numStochasticSamples=0, size_t maxNumIterations=100000, double tolerance=1e-8);
    virtual void evaluateIncrements();
    virtual void solve(const Matrix& X, const Vector& y);
    virtual void solve(const Matrix& X, const std::vector<bool>& yB);
    Vector getProbability(const Matrix& X) const;
    double getProbability(const Vector& xrow) const;
    virtual Vector predict(const Matrix& X) const;
    virtual double predict(const Vector& xrow) const;
    virtual bool predictB(const Vector& xrow) const;
    virtual std::vector<bool> predictB(const Matrix& X) const;
};

#endif