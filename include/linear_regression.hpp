#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "vectr.hpp"

class LinearRegressionSolver
{
    Vector m_weights;
    double m_bias;
    void solveAnalytical(const Matrix& X, const Vector& y);
    void solveGradientDescent(const Matrix& X, const Vector& y);
    void solveStochasticGradientDescent(const Matrix& X, const Vector& y);
public:
    enum SolutionMethodEnum {ANALYTICAL, GRADIENT_DESCENT, STOCHASTIC_GRADIENT_DESCENT};
    LinearRegressionSolver();
    void solve(const Matrix& X, const Vector& y, int solutionMethod=SolutionMethodEnum::ANALYTICAL);
    const Vector& getWeights() const;
    double getBias() const;
};

#endif