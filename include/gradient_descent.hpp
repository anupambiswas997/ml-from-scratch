#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include "vectr.hpp"

// Gradient Descent is an iterative process. Each iteration of the process
// may be summarized to be have the following:
//   a. compute gradients of the cost function w.r.t.weights and bias
//   b. update the weights and bias using the computed gradients and hyperparameters
//   c. inspect if the conditions to discontinue the iterations have been achieved.
// In the above steps, apart from step (a), other steps are quite generic. Step (a)
// varies from problem to problem. For example, the method of computing the gradients
// for Linear Regression and Logistic Regression can be different, as the former may
// use ordinary least squares, and the latter may use log-loss.
// To avoid having to repeat writing the logic for the generic parts, the task of
// computing the gradients, and hence the increments of weights and bias, may be
// delegated to a different class, here called IGDIncrementsEvaluator. Using this
// interface class, only the variable step (a) will need to be written for different
// models that require Gradient Descent solution to obtain weights and bias.

class IGDIncrementsEvaluator
{
protected:
    Vector m_weightIncrements;
    double m_biasIncrement;
public:
    virtual void evaluateIncrements() = 0;
    virtual const Vector& getWeightIncrements()
    {
        return m_weightIncrements;
    }
    virtual double getBiasIncrement()
    {
        return m_biasIncrement;
    }
};

void solveGradientDescent(IGDIncrementsEvaluator *gdincEvaluator, size_t numWeights, double tolerance=1.0e-8, size_t maxNumIterations=100000);

#endif