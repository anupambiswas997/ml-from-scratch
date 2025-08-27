#include "linear_regression.hpp"
#include <iostream>
#include "matrix.hpp"
#include "sparse_matrix.hpp"
#include "sparse_vector.hpp"
#include "random_quantities.hpp"
#include <cmath>

Indexer::Indexer(size_t size, bool doShuffle)
{
    m_indices = {};
    for(size_t i = 0; i < size; i++)
    {
        m_indices.push_back(i);
    }
    m_doShuffle = doShuffle;
}

void Indexer::update()
{
    if(!m_doShuffle)
    {
        return;
    }
    for(size_t i = 0; i < m_indices.size(); i++)
    {
        size_t i2 = rand() % m_indices.size();
        size_t temp = m_indices[i];
        m_indices[i] = m_indices[i2];
        m_indices[i2] = temp;
    }
}

size_t Indexer::getIndex(size_t i)
{
    return m_indices[i];
}

LinearRegressionSolver::LinearRegressionSolver()
{
    std::cout << "Created LinearRegressionSolver object" << std::endl;
}

void LinearRegressionSolver::solveAnalytical(const Matrix& X, const Vector& y)
{
    size_t m = X.getNumRows();
    double oneByM = 1.0 / m;
    Matrix XT = X.getTranspose();
    SparseMatrix iden(0, m, m);
    for(size_t i = 0; i < m; i++)
    {
        iden[i][i] = 1;
    }
    SparseMatrix U = iden - oneByM;
    Matrix XTU = XT * U;
    Matrix XTUX = XTU * X;
    Matrix XTUXinverse = XTUX.getInverse();
    Vector XTUy = XTU * y;
    m_weights = XTUXinverse * XTUy;
    double sum = 0;
    Vector Xw = X * m_weights;
    for(size_t i = 0; i < m; i++)
    {
        sum += (y[i] - Xw[i]);
    }
    m_bias = oneByM * sum;
}

void LinearRegressionSolver::solveGradientDescent(const Matrix& X, const Vector& y, size_t numStochasticSamples, double learningRate, double tolerance, size_t maxNumIterations)
{
    // Initialize bias and weights.
    m_bias = getRandom();
    m_weights = getRandomVector(X.getNumColumns());

    // Variables that will determine whether or not to continue iterating:
    bool cond = true;
    size_t iterationCount = 0;
    double maxIncrement;
    bool incrementCond;
    bool iterCond;

    // Gradient descent is performed with:
    //   * entire X, when isStochastic = false
    //   * randomly chosen numStochasticSamples entries of X, when isStochastic = true
    bool isStochastic = (numStochasticSamples > 0);

    // Indexer object helps to:
    //   * randomly select numStochasticSamples entries from X for stochastic GD (isStochastic = true)
    //   * just get all entries in the case of batch GD (isStochastic = false)
    Indexer indexer(X.getNumRows(), isStochastic);

    // For partial derivative calculations in every iteration of GD, the number
    // of data entries (numRows) to take into account depends on whether stochastic or batch GD
    // is being performed.
    size_t numRows = isStochastic ? numStochasticSamples : X.getNumRows();
    size_t numColumns = X.getNumColumns();
    double mInv = 1.0 / numRows;
    double mInvNegLR = -mInv * learningRate;

    while(cond)
    {
        // If this is stochastic gradient descent, then calling update will
        // shuffle the rows of the indexer object, allowing for a random but smaller
        // selection of data entries, for this iteration's calculations. Otherwise,
        // calling update will not affect the indexer object, i.e. calling
        // indexer.getIndex(i) will return i;
        indexer.update();
        // error vector
        std::vector<double> err = {};
        for(size_t i = 0; i < numRows; i++)
        {
            size_t iActual = indexer.getIndex(i);
            double sum = m_bias - y[iActual];
            for(size_t j = 0; j < numColumns; j++)
            {
                sum += X.getData()[iActual][j] * m_weights.getData()[j];
            }
            err.push_back(sum);
        }
        std::vector<double> dCdwVec = {};
        for(size_t i = 0; i < numColumns; i++)
        {
            double sum = 0;
            for(size_t j = 0; j < numRows; j++)
            {
                // sum += (X-transpose[i][indexer.getIndex(j)] times err[j])
                // but X-transpose[i][indexer.getIndex(j)] = X[indexer.getIndex(j)][i]
                sum += (X.getData()[indexer.getIndex(j)][i] * err[j]);
            }
            dCdwVec.push_back(sum);
        }

        // Partial derivative of cost function w.r.t. weights: dCdw = Xsam-transpose * err / numRows
        // Xsam -> entire X or sampled X in case of stochastic gradient descent
        // Increment in weights for gradient descent: dCdw * negative-learning-rate
        Vector weightsIncrement = Vector(dCdwVec) * mInvNegLR;

        // Partial derivative of cost function w.r.t. bias: dCdb = SUM(err) / numRows
        // Increment in bias for gradiant descent: dCdb * negative-learning-rate
        double biasIncrement = Vector(err).getSum() * mInvNegLR;

        m_weights = m_weights + weightsIncrement;
        m_bias += biasIncrement;

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
    std::cout << "    Linear regression gradient descent finished with:" << std::endl;
    std::cout << "    Number of iterations completed: " << iterationCount << std::endl;
    std::cout << "    Maximum number of iterations: " << maxNumIterations << std::endl;
    std::cout << "    Last absolute weights/bias increment: " << maxIncrement << std::endl;
    std::cout << "    Error tolerance used: " << tolerance << std::endl;
}

void LinearRegressionSolver::solve(const Matrix& X, const Vector& y, int solutionMethod)
{
    if(solutionMethod == ANALYTICAL)
    {
        solveAnalytical(X, y);
    }
    else if(solutionMethod == BATCH_GRADIENT_DESCENT)
    {
        solveGradientDescent(X, y);
    }
    else if(solutionMethod == STOCHASTIC_GRADIENT_DESCENT)
    {
        solveGradientDescent(X, y, size_t(0.5 * X.getNumRows()));
    }
    else
    {
        std::cout << "*** Incorrect solution method: " << solutionMethod << std::endl;
    }
}

const Vector& LinearRegressionSolver::getWeights() const
{
    return m_weights;
}

double LinearRegressionSolver::getBias() const
{
    return m_bias;
}