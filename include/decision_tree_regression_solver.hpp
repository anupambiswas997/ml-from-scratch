#ifndef DECISION_TREE_REGRESSION_SOLVER_HPP
#define DECISION_TREE_REGRESSION_SOLVER_HPP

#include "base_solver.hpp"

struct DecisionTree
{
    size_t column;
    double splitValue;
    double value;
    bool isLeaf;
    DecisionTree* left;
    DecisionTree* right;

    DecisionTree();
    double getValue(const Vector& x) const;
};

// DecisionTreeRegressionSolver is the class that can be used
// to create a decision tree for a regression problem.

class DecisionTreeRegressionSolver: virtual public BaseSolver
{
    // Leaf nodes should not have more elements than this number.
    size_t m_maxLeafSize;
    DecisionTree* m_tree;
    void buildDecisionTree(const Matrix& X, const Vector& y, const std::vector<size_t>& indicesToInspect);
public:
    DecisionTreeRegressionSolver(size_t maxLeafSize=5);
    virtual void solve(const Matrix& X, const Vector& y);
    virtual Vector predict(const Matrix& X) const;
    virtual double predict(const Vector& xrow) const;
};

#endif