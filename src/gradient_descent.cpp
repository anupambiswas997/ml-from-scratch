#include "gradient_descent.hpp"
#include "random_quantities.hpp"
#include <cmath>

void solveGradientDescent(IGDIncrementsEvaluator *gdincEvaluator, size_t numWeights, double tolerance, size_t maxNumIterations)
{
    // Initialize bias and weights.
    double bias = getRandom();
    Vector weights = getRandomVector(numWeights);

    // Variables that will determine whether or not to continue iterating:
    bool cond = true;
    size_t iterationCount = 0;
    double maxIncrement;
    bool incrementCond;
    bool iterCond;

    // Gradient descent is performed with:
    //   * entire X, when isStochastic = false
    //   * randomly chosen numStochasticSamples entries of X, when isStochastic = true
    //bool isStochastic = (numStochasticSamples > 0);

    // Indexer object helps to:
    //   * randomly select numStochasticSamples entries from X for stochastic GD (isStochastic = true)
    //   * just get all entries in the case of batch GD (isStochastic = false)
    //Indexer indexer(X.getNumRows(), isStochastic);

    // For partial derivative calculations in every iteration of GD, the number
    // of data entries (numRows) to take into account depends on whether stochastic or batch GD
    // is being performed.
    //size_t numRows = isStochastic ? numStochasticSamples : X.getNumRows();
    //size_t numColumns = X.getNumColumns();
    //double mInv = 1.0 / numRows;
    //double mInvNegLR = -mInv * learningRate;

    while(cond)
    {
        gdincEvaluator->evaluateIncrements();
        const Vector& weightsIncrement = gdincEvaluator->getWeightIncrements();
        double biasIncrement = gdincEvaluator->getBiasIncrement();
        weights = weights + weightsIncrement;
        bias += biasIncrement;

        // Compute maximum error
        double absMinWIncrement = fabs(weightsIncrement.getMin());
        double absMaxWIncrement = fabs(weightsIncrement.getMax());
        double wIncrement = (absMinWIncrement < absMaxWIncrement) ? absMinWIncrement : absMaxWIncrement;
        double absBiasIncrement = fabs(biasIncrement);
        maxIncrement = (wIncrement < absBiasIncrement) ? absBiasIncrement : wIncrement;

        iterationCount++;
        incrementCond = maxIncrement > tolerance;
        iterCond = iterationCount < maxNumIterations;
        // The looping to find optimal weights continues if:
        //   a. the maximum error is still greater than the provided tolerance, and
        //   b. the number of iterations is smaller than the provided iteration limit.
        cond = incrementCond && iterCond;
    }
    /*
    std::cout << "    Linear regression gradient descent finished with:" << std::endl;
    std::cout << "    Number of iterations completed: " << iterationCount << std::endl;
    std::cout << "    Maximum number of iterations: " << maxNumIterations << std::endl;
    std::cout << "    Last absolute weights/bias increment: " << maxIncrement << std::endl;
    std::cout << "    Error tolerance used: " << tolerance << std::endl;//*/
}