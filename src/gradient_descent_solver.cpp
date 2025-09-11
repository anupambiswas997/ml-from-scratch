#include "gradient_descent_solver.hpp"
#include "random_quantities.hpp"
#include <cmath>
#include "matrix.hpp"

GradientDescentSolver::GradientDescentSolver(size_t numIterations, double tolerance)
{
    m_maxIterations = numIterations;
    m_tolerance = tolerance;
}

bool GradientDescentSolver::shouldContinueIterating()
{
    // Compute maximum error
    double absMinWIncrement = fabs(m_weightIncrements.getMin());
    double absMaxWIncrement = fabs(m_weightIncrements.getMax());
    double wIncrement = (absMinWIncrement < absMaxWIncrement) ? absMinWIncrement : absMaxWIncrement;
    double absBiasIncrement = fabs(m_biasIncrement);
    m_maxIncrement = (wIncrement < absBiasIncrement) ? absBiasIncrement : wIncrement;

    bool incrementCond = m_maxIncrement > m_tolerance;
    bool iterCond = m_iterationCount < m_maxIterations;
    // The looping to find optimal weights continues if:
    //   a. the maximum error is still greater than the provided tolerance, and
    //   b. the number of iterations is smaller than the provided iteration limit.
    return incrementCond && iterCond;
}

void GradientDescentSolver::log() const
{
    // Nothing to log, by default
}

void GradientDescentSolver::solve(const Matrix& X, const Vector& y)
{
    // Initialize bias and weights.
    m_bias = getRandom();
    m_weights = getRandomVector(X.getNumColumns());

    // Variables that will determine whether or not to continue iterating:
    bool cond = true;
    m_iterationCount = 0;

    while(cond)
    {
        evaluateIncrements();
        m_weights = m_weights + m_weightIncrements;
        m_bias += m_biasIncrement;
        m_iterationCount++;
        cond = shouldContinueIterating();
        log();
    }
}