#ifndef DECISION_TREE_REGRESSION_SOLVER_HPP
#define DECISION_TREE_REGRESSION_SOLVER_HPP

#include "base_solver.hpp"

// DecisionTree is a tree data structure and the forms the core
// of the decision tree solution process. Unlike weights and bias terms
// in other machine learning models, the decision tree solver constructs
// a tree data structure from the training data.
// A DecisionTree object will either be:
//   - an internal node which will branch into left and right nodes; or
//   - a leaf node which will NOT branch into left and right nodes.
// An internal node is associated with a condition based on:
//   - a particular column of input vector; and
//   - a split-value.
// Basically the internal node's condition is as follows:
// if input-vector[column] < split-value, follow left node, otherwise right.
// A leaf node is a terminal node and is associated with a value, which is
// used as the target for the input-vector.

struct DecisionTree
{
    size_t column;
    double splitValue;
    double value;
    bool isLeaf;
    DecisionTree* left;
    DecisionTree* right;

    DecisionTree();
    ~DecisionTree();
    double getValue(const Vector& x) const;
    std::string getText() const;
    void describe(std::string indent="") const;
};

// DecisionTreeRegressionSolver is the class that can be used
// to create a decision tree for a regression problem. Basically
// this solver contains a DecisionTree object which is constructed
// during the training process.
// Although decision tree solvers can have different types of
// criteria for the leaf nodes, i.e. to stop further branching, this
// particular implementation is based on the criterion of maximum leaf
// size. That implies, there is a limit on the maximum number of input
// vectors from which a leaf node's value should be determined.
// In other words, during the training process, when a node is being
// constructed with a subset of the training data, if the size of the
// subset is greater than the maximum leaf size, the node gets to have
// left and right branching. Otherwise, it is associated with a target
// value equal to the mean of the target values associated with subset. 

class DecisionTreeRegressionSolver: virtual public BaseSolver
{
    // Leaf nodes should not have more elements than this number.
    size_t m_maxLeafSize;
    DecisionTree* m_tree;
    size_t m_nodeCount;
    bool m_verbose;

    void buildDecisionTree(const Matrix& X, const Vector& y, const std::vector<size_t>& indicesToInspect, DecisionTree* tree);
public:
    DecisionTreeRegressionSolver(size_t maxLeafSize=5, bool verbose=false);
    ~DecisionTreeRegressionSolver();
    size_t getNodeCount() const;
    void describeTree() const;
    virtual void solve(const Matrix& X, const Vector& y);
    virtual Vector predict(const Matrix& X) const;
    virtual double predict(const Vector& xrow) const;
};

#endif