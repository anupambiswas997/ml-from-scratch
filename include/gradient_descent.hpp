#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include "vectr.hpp"

class IGradientEvaluator
{
protected:
    Vector m_weightIncrements;
    double m_biasIncrement;
public:
    virtual void evaluateGradientAndIncrements() = 0;
    virtual const Vector& getWeightIncrements();
    virtual double getBiasIncrement();
};

void solveGradientDescent(IGradientEvaluator *gradEvaluator, size_t numWeights, double tolerance=1.0e-8, size_t maxNumIterations=100000);

#endif