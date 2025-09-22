#include "decision_tree_regression_solver.hpp"
#include "indexing_utils.hpp"
#include <cassert>
#include <iostream>
#include <sstream>

DecisionTree::DecisionTree()
{
    isLeaf = true;
    left = 0;
    right = 0;
}

DecisionTree::~DecisionTree()
{
    delete left;
    delete right;
}

double DecisionTree::getValue(const Vector& x) const
{
    if(isLeaf)
    {
        return value;
    }
    return (x.getData()[column] < splitValue) ? left->getValue(x) : right->getValue(x);
}

std::string DecisionTree::getText() const
{
    std::ostringstream ss;
    if(isLeaf)
    {
        ss << "LEAF:VALUE " << value;
    }
    else
    {
        ss << "INTERNAL:SPLIT-CONDITION x[" << column << "] < " << splitValue;
    }
    return ss.str();
}

DecisionTreeRegressionSolver::DecisionTreeRegressionSolver(size_t maxLeafSize, bool verbose)
{
    m_tree = 0;
    m_nodeCount = 0;
    m_maxLeafSize = maxLeafSize;
    m_verbose = verbose;
}

DecisionTreeRegressionSolver::~DecisionTreeRegressionSolver()
{
    if(m_verbose)
    {
        std::cout << "Deleting tree" << std::endl;
    }
    delete m_tree;
}

size_t DecisionTreeRegressionSolver::getNodeCount() const
{
    return m_nodeCount;
}

void DecisionTree::describe(std::string indent) const
{
    std::cout << indent << getText() << std::endl;
    if(left)
    {
        left->describe(indent + "   ");
    }
    if(right)
    {
        right->describe(indent + "   ");
    }
}

void DecisionTreeRegressionSolver::describeTree() const
{
    if(m_tree)
    {
        m_tree->describe();
    }
}

std::pair<size_t, double> getOptimalSplit(const Vector& y, size_t column, const std::vector<size_t>& sortedIndices, double ysum)
{
    double minRSSVal = 0;
    size_t index;
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

void DecisionTreeRegressionSolver::buildDecisionTree(const Matrix& X, const Vector& y, const std::vector<size_t>& indicesToInspect, DecisionTree *tree)
{
    size_t curNodeId = m_nodeCount;
    if(m_verbose)
    {
        std::cout << "Building tree node: " << curNodeId << std::endl;
    }
    m_nodeCount++;
    assert(tree != 0);
    double ysum = 0;
    for(const auto& i: indicesToInspect)
    {
        ysum += y[i];
    }
    if(indicesToInspect.size() <= m_maxLeafSize)
    {
        tree->value = ysum / indicesToInspect.size();
        tree->isLeaf = true;
        if(m_verbose)
        {
            std::cout << "  -> Node " << curNodeId << ": " << tree->getText() << std::endl;
        }
        return;
    }
    double minRSSVal = 0;
    size_t optimalColumn;
    size_t optimalIndex;
    std::vector<size_t> optimalIndices;
    for(size_t j = 0; j < X.getNumColumns(); j++)
    {
        std::vector<size_t> sortedIndices = getColumnSortedIndices(X, j, indicesToInspect);
        std::pair<size_t, double> split = getOptimalSplit(y, j, sortedIndices, ysum);
        if(split.second < minRSSVal)
        {
            minRSSVal = split.second;
            optimalIndex = split.first;
            optimalColumn = j;
            optimalIndices = sortedIndices;
        }
    }
    std::vector<size_t> leftIndices = {};
    std::vector<size_t> rightIndices = {};
    for(size_t i = 0; i < optimalIndices.size(); i++)
    {
        std::vector<size_t>& leftOrRightIndices = (i < optimalIndex) ? leftIndices : rightIndices;
        leftOrRightIndices.push_back(i);
    }
    tree->isLeaf = false;
    tree->column = optimalColumn;
    tree->splitValue = X.getData()[optimalIndices[optimalIndex]][optimalColumn];
    DecisionTree *left = new DecisionTree();
    DecisionTree *right = new DecisionTree();
    tree->left = left;
    tree->right = right;
    if(m_verbose)
    {
        std::cout << "  -> Node " << curNodeId << ": " << tree->getText() << std::endl;
    }
    buildDecisionTree(X, y, leftIndices, left);
    buildDecisionTree(X, y, rightIndices, right);
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
    m_tree = new DecisionTree();
    buildDecisionTree(X, y, indices, m_tree);
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