#include "logistic_regression_solver.hpp"
#include "ml_functions.hpp"

LogisticRegressionSolver::LogisticRegressionSolver(double learningRate, size_t numStochasticSamples, size_t maxNumIterations, double tolerance)
:IGradientDescentSolver(maxNumIterations, tolerance),
GradientDescentData(numStochasticSamples, learningRate)
{
}

void LogisticRegressionSolver::evaluateIncrements()
{
    m_indexer.update();

    // error vector
    std::vector<double> err = {};
    for(size_t i = 0; i < m_numRows; i++)
    {
        size_t iActual = m_indexer.getIndex(i);
        double sum = m_bias;// - m_py->getData()[iActual];
        for(size_t j = 0; j < m_numColumns; j++)
        {
            sum += m_pX->getData()[iActual][j] * m_weights.getData()[j];
        }
        err.push_back(m_py->getData()[iActual] - sigmoid(sum));
    }
    std::vector<double> dCdwVec = {};
    for(size_t i = 0; i < m_numColumns; i++)
    {
        double sum = 0;
        for(size_t j = 0; j < m_numRows; j++)
        {
            // sum += (X-transpose[i][indexer.getIndex(j)] times err[j])
            // but X-transpose[i][indexer.getIndex(j)] = X[indexer.getIndex(j)][i]
            sum += (m_pX->getData()[m_indexer.getIndex(j)][i] * err[j]);
        }
        dCdwVec.push_back(sum);
    }

    // Partial derivative of cost function w.r.t. weights: dCdw = Xsam-transpose * err / numRows
    // Xsam -> entire X or sampled X in case of stochastic gradient descent
    // Increment in weights for gradient descent: dCdw * negative-learning-rate
    m_weightIncrements = Vector(dCdwVec) * m_constMult;

    // Partial derivative of cost function w.r.t. bias: dCdb = SUM(err) / numRows
    // Increment in bias for gradiant descent: dCdb * negative-learning-rate
    m_biasIncrement = Vector(err).getSum() * m_constMult;
}

void LogisticRegressionSolver::solve(const Matrix& X, const Vector& y)
{
    setData(X, y);
    IGradientDescentSolver::solve(X, y);
}

Vector LogisticRegressionSolver::getProbability(const Matrix& X) const
{
    Vector temp = SimpleSolver::predict(X);
    std::vector<double> res = {};
    for(size_t i = 0; i < X.getNumRows(); i++)
    {
        res.push_back(sigmoid(temp.getData()[i]));
    }
    return Vector(res);
}

double LogisticRegressionSolver::getProbability(const Vector& xrow) const
{
    return sigmoid(SimpleSolver::predict(xrow));
}

Vector LogisticRegressionSolver::predict(const Matrix& X) const
{
    Vector temp = getProbability(X);
    std::vector<double> res = {};
    for(size_t i = 0; i < X.getNumRows(); i++)
    {
        bool isOne = (temp.getData()[i] > 0.5);
        res.push_back(isOne ? 1 : 0);
    }
    return Vector(res);
}

double LogisticRegressionSolver::predict(const Vector& xrow) const
{
    return (getProbability(xrow) > 0.5) ? 1 : 0;
}