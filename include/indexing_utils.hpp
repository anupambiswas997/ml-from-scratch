#ifndef INDEXING_UTILS_HPP
#define INDEXING_UTILS_HPP

#include "matrix.hpp"

// Rationale:
// For producing decision trees, subsets of the training dataset needs
// to be sorted according to different feature values. Rearranging the dataset
// may be expensive. It may be more useful to have a rearranged index list,
// such that, when a particular feature's values are viewed according to the list,
// the feature values are observed to be in ascending order.

// ColumnSortFunctor is the functor that provides the operator method
// to determine how indices should be sorted. Default implementation will
// place the smaller value before the greater value, as that will be used
// for resolving decision trees.

class ColumnSortFunctor
{
protected:
    // m_pX: the dataset of which the row indices need to be rearranged
    const Matrix* m_pX;
    // column: the dataset column of which the values will be used to obtained
    // the rearranged index list
    size_t m_column;
public:
    ColumnSortFunctor(const Matrix& X, size_t column);
    virtual bool operator()(const size_t& i, const size_t& j);
};

// getColumnSortedIndices returns indices according to the ascending
// order values of a column of the dataset. At a particular point, not all rows
// of the dataset needs to be inspected for this sorting, but a selected (from a
// previous sort) list of indices.
// X: dataset for which rearranged indices are to be obtained
// column: the column ID of the dataset of which the values are to be used
// indicesToInspect: the indices of the dataset to look into

std::vector<size_t> getColumnSortedIndices(const Matrix& X, size_t column, const std::vector<size_t>& indicesToInspect);

#endif