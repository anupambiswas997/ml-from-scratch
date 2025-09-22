#include "decision_tree_regression_solver.hpp"
#include "indexing_utils.hpp"
#include <cassert>

DecisionTree::DecisionTree()
{
    isLeaf = true;
    left = 0;
    right = 0;
}

double DecisionTree::getValue(const Vector& x) const
{
    if(isLeaf)
    {
        return value;
    }
    return (x.getData()[column] < splitValue) ? left->getValue(x) : right->getValue(x);
}

DecisionTreeRegressionSolver::DecisionTreeRegressionSolver(size_t maxLeafSize)
{
    m_tree = 0;
    m_maxLeafSize = maxLeafSize;
}

std::pair<size_t, double> getOptimalSplit(const Vector& y, size_t column, const std::vector<size_t>& sortedIndices)
{
    double minRSSVal = 0;
    size_t index;
    double ysum = 0;
    for(size_t i = 0; i < sortedIndices.size(); i++)
    {
        ysum += y[sortedIndices[i]];
    }
    double xa = 0;
    for(size_t i = 1; i < sortedIndices.size(); i++)
    {
        xa = (xa * (i - 1) + y[sortedIndices[i]]) / i;
        double xb = (ysum - i * xa) / (sortedIndices.size() - i);
        double curRSSVal = -(i * xa * xa + (sortedIndices.size() - i) * xb * xb);
        if(curRSSVal < minRSSVal)
        {
            minRSSVal = curRSSVal;
            index = i;
        }
    }
    return std::make_pair(index, minRSSVal);
}

void DecisionTreeRegressionSolver::buildDecisionTree(const Matrix& X, const Vector& y, const std::vector<size_t>& indicesToInspect)
{
    double minRSSVal = 0;
    size_t optimalColumn;
    size_t optimalIndex;
    std::vector<size_t> optimalIndices;
    for(size_t j = 0; j < X.getNumColumns(); j++)
    {
        std::vector<size_t> sortedIndices = getColumnSortedIndices(X, j, indicesToInspect);
        std::pair<size_t, double> split = getOptimalSplit(y, j, sortedIndices);
        if(split.second < minRSSVal)
        {
            minRSSVal = split.second;
            optimalIndex = split.first;
            optimalColumn = j;
            optimalIndices = sortedIndices;
        }
    }
}

void DecisionTreeRegressionSolver::solve(const Matrix& X, const Vector& y)
{
    assert(y.size() > 0);
    assert(y.size() == X.getNumRows());
    std::vector<size_t> indices = {};
    for(size_t i = 0; i < y.size(); i++)
    {
        indices.push_back(i);
    }
    buildDecisionTree(X, y, indices);
}

Vector DecisionTreeRegressionSolver::predict(const Matrix& X) const
{
    std::vector<double> r = {};
    for(size_t i = 0; i < X.getNumRows(); i++)
    {
        r.push_back(predict(Vector(X.getData()[i])));
    }
    return Vector(r);
}

double DecisionTreeRegressionSolver::predict(const Vector& xrow) const
{
    return m_tree->getValue(xrow);
}